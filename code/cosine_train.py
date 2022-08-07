from language import detect_nb_lang
from common import clean_html
import random
import torch
import numpy as np
from torch.optim import AdamW
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from state import State
from common import get_markdown_cells, get_code_cells
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from wandb_helper import init_wandb

end_token = 'END'


@dataclass
class MiniBatch:
    markdowns: list
    code: list
    correct_idx: list  # for each markdown store idx in code
    max_len_cache: int

    def append(self, cur_markdown, cur_code):
        self.markdowns.append(cur_markdown)
        if cur_code in self.code:
            self.correct_idx.append(self.code.index(cur_code))
        else:
            self.code.append(cur_code)
            self.correct_idx.append(len(self.code) - 1)

    def get_max_len(self):
        if self.max_len_cache == 0:
            texts_len = [len(t) for t in (self.markdowns + self.code)]
            self.max_len_cache = max(texts_len)
        return self.max_len_cache

    def cnt(self):
        return len(self.markdowns) + len(self.code)


@dataclass
class Batch:
    mini: list
    sum_cnt: int

    def append(self, mini_batch):
        self.mini.append(mini_batch)
        self.sum_cnt += mini_batch.cnt()

    def get_all_texts(self):
        all = []
        for mini in self.mini:
            all += mini.markdowns
            all += mini.code
        return all


@dataclass
class Sample:
    markdown: str
    code: str


def gen_batches(state: State, next_code_cells_cnt, sep_token, rand_seed):
    random.seed(rand_seed)

    df = state.cur_train_nbs
    all = df.index.get_level_values(0).unique()
    print('Total nbs:', len(all))

    minibatches = []
    for nb_id in tqdm(all):
        nb = df.loc[nb_id]

        lang = detect_nb_lang(nb)

        correct_order = state.df_orders.loc[nb_id]
        correct_order.append(end_token)
        markdown_cell_ids = get_markdown_cells(nb)

        def get_code(cell_id):
            if cell_id == end_token:
                return end_token
            return nb.loc[cell_id]['source']

        def get_codes(cell_ids):
            res = get_code(cell_ids[0])
            for cell in cell_ids[1:]:
                res += sep_token + get_code(cell)
            return res

        def transform_markdown(text):
            text = clean_html(text)
            if lang != 'en':
                text = state.easymnt.translate(text, target_lang='en')
            return text

        samples = []
        for pos, cell_id in enumerate(correct_order):
            if cell_id in markdown_cell_ids:
                next_code_cells = []
                for next_cell in correct_order[pos:]:
                    if next_cell not in markdown_cell_ids:
                        next_code_cells.append(next_cell)
                        if len(next_code_cells) == next_code_cells_cnt:
                            break
                assert len(next_code_cells) != 0
                samples.append(
                    Sample(markdown=transform_markdown(nb.loc[cell_id]['source']), code=get_codes(next_code_cells)))
        random.shuffle(samples)
        num_chunks = (len(samples) + state.config.cosine_minibatch_size -
                      1) // state.config.cosine_minibatch_size

        for batch_samples in np.array_split(samples, num_chunks):
            batch = MiniBatch(markdowns=[], code=[],
                              correct_idx=[], max_len_cache=0)
            for sample in batch_samples:
                batch.append(sample.markdown, sample.code)
            minibatches.append(batch)
    print('Sorting minibatches')
    minibatches.sort(key=lambda x: x.get_max_len())
    print('Done sorting minibatches')

    batches = []
    for b in minibatches:
        if len(batches) == 0 or batches[-1].sum_cnt + b.cnt() > state.config.cosine_batch_size:
            batches.append(Batch(mini=[], sum_cnt=0))
        batches[-1].append(b)

    random.shuffle(batches)
    return batches


def train_on_batch(state: State, batch, model):
    all_texts = batch.get_all_texts()
    encoded = model.encode_texts(state, all_texts)
    embeddings = model(
        input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], use_sigmoid=False, return_scalar=False)

    markdown_vec = []
    code_vec = []
    expected_order = []

    shift = 0
    code_shift = 0

    for mini in batch.mini:
        markdown_vec += embeddings[shift:shift+len(mini.markdowns)]
        code_vec += embeddings[shift+len(mini.markdowns):shift+mini.cnt()]
        shift += mini.cnt()
        expected_order += [(x + code_shift) for x in mini.correct_idx]
        code_shift += len(mini.code)

    scores = torch.einsum("ab,cb->ac", torch.stack(markdown_vec),
                          torch.stack(code_vec)) * model.coef_mul

    expected_order = torch.tensor(expected_order).to(state.device)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(scores, expected_order)
    loss.backward()
    return loss.item()


def run_train_all_new(state: State, model, rand_seed=787788, optimizer_state=None):
    print('Start training')
    print('Start generating batches...')
    batches = gen_batches(
        state, next_code_cells_cnt=model.next_code_cells, sep_token=model.tokenizer.sep_token, rand_seed=rand_seed)
    print('Generated batches:', len(batches))

    model.zero_grad()
    model.train()

    learning_rate = 3e-5
    grad_batch_sz = 1
    steps = len(batches) // grad_batch_sz

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    if optimizer_state is not None:
        print('loading optimizer state...')
        optimizer.load_state_dict(torch.load(optimizer_state))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    init_wandb(name="train-cosine-next-c-s=" + str(model.next_code_cells))

    last_loss = 0.0

    sum_batch_losses = 0.0
    cnt_batch_losses = 0

    for id, batch in enumerate(tqdm(batches)):
        cur_loss = train_on_batch(state, batch, model)

        sum_batch_losses += cur_loss
        cnt_batch_losses += 1
        if cnt_batch_losses == grad_batch_sz:
            last_loss = sum_batch_losses / cnt_batch_losses
            sum_batch_losses = 0.0
            cnt_batch_losses = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if id > 10:
            wandb.log(
                {'loss': last_loss, 'learning_rate': scheduler.get_last_lr()[0]})

        if (id % 10000 == 9999):
            print('Saving model after', id)
            model.save('3graph-batch-' + str(id), optimizer=optimizer)

    wandb.finish()
    model.save('graph3-cur-final', optimizer=optimizer)
    # TODO: save model/optimizer/scheduler
