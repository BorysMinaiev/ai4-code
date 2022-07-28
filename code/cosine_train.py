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

default_mul = 10
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


def gen_batches(state: State):
    random.seed(787788)

    df = state.cur_train_nbs
    all = df.index.get_level_values(0).unique()
    print('Total nbs:', len(all))

    minibatches = []
    for nb_id in tqdm(all):
        nb = df.loc[nb_id]
        correct_order = state.df_orders.loc[nb_id]
        correct_order.append(end_token)
        markdown_cell_ids = get_markdown_cells(nb)

        def get_code(cell_id):
            if cell_id == end_token:
                return end_token
            return nb.loc[cell_id]['source']

        samples = []
        for pos, cell_id in enumerate(correct_order):
            if cell_id in markdown_cell_ids:
                next_code_cell = None
                for next_cell in correct_order[pos:]:
                    if next_cell not in markdown_cell_ids:
                        next_code_cell = next_cell
                        break
                assert next_code_cell != None
                samples.append(
                    Sample(markdown=nb.loc[cell_id]['source'], code=get_code(next_code_cell)))
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


def train_on_batch(state: State, batch, model, optimizer, scheduler):
    all_texts = batch.get_all_texts()
    encoded = model.encode_texts(state, all_texts)
    embeddings = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], use_sigmoid=False, return_scalar=False)

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
                          torch.stack(code_vec)) * default_mul

    expected_order = torch.tensor(expected_order).to(state.device)

    loss_fct = CrossEntropyLoss()
    # print('scores:', scores)
    # print('expected order:', expected_order)
#     for i in range(len(expected_order)):
#         print('expected:', expected_order[i])
#         for val in scores[i]:
#             print('%.3f ' % val, end='')
#         print()
    
    loss = loss_fct(scores, expected_order)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    return loss.item()


def run_train_all_new(state: State, model):
    print('Start training')
    print('Start generating batches...')
    batches = gen_batches(state)
    print('Generated batches:', len(batches))

    model.zero_grad()
    model.train()

    learning_rate = 3e-5
    steps = len(batches)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    init_wandb(name="train-cosine")

    for id, batch in enumerate(tqdm(batches)):
        cur_loss = train_on_batch(state, batch, model, optimizer, scheduler)
        wandb.log({'loss': cur_loss})

    wandb.finish()
    # TODO: save model/optimizer/scheduler
