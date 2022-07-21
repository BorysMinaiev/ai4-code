import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import wandb
from wandb_helper import init_wandb
from tqdm import tqdm
from dataclasses import dataclass

from common import split_into_batches
from state import State

import torch

max_tokenizer_len = 512


@dataclass
class LearnSample:
    text: str
    relative_position: float
    code_texts: list


class MyRobertaModel(nn.Module):
    def __init__(self, preload_state=None):
        super(MyRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.top = nn.Linear(768, 1)
        if preload_state is not None:
            print('Preloading state:', preload_state)
            self.load_state_dict(torch.load(preload_state))
        self.name = preload_state if preload_state is not None else "0"

    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = self.top(x[:, 0, :])
        return x

    def encode_one(self, text, max_length):
        return self.tokenizer.batch_encode_plus(
            [text],
            add_special_tokens=False,
            max_length=max_length,
            padding="longest",
            truncation=True,
            pad_to_max_length=True
        )['input_ids'][0]

    def encode_sample(self, sample: LearnSample):
        input_ids = [self.tokenizer.cls_token_id] + \
            self.encode_one(sample.text, max_length=64) + \
            [self.tokenizer.sep_token_id]
        filtered_code_texts = []
        cnt_texts = 20
        if len(sample.code_texts) <= cnt_texts:
            filtered_code_texts = sample.code_texts
        else:
            prev_pos = -1
            for i in range(cnt_texts - 1):
                pos = i * len(sample.code_texts) // cnt_texts
                assert pos != prev_pos
                prev_pos = pos
                filtered_code_texts.append(sample.code_texts[pos])
            filtered_code_texts.append(sample.code_texts[-1])

        for text in filtered_code_texts:
            input_ids.extend(self.encode_one(text, max_length=21))
            input_ids.append(self.tokenizer.sep_token_id)

        assert len(input_ids) <= max_tokenizer_len

        cur_len = len(input_ids)
        attention_mask = [1] * cur_len + [0] * (max_tokenizer_len - cur_len)
        input_ids.extend([self.tokenizer.pad_token_id] *
                         (max_tokenizer_len - cur_len))

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def encode(self, state: State, samples):
        input_ids = []
        attention_mask = []

        for sample in samples:
            encoded = self.encode_sample(sample)
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])

        # print('input  ids:', input_ids)

        return {'input_ids': torch.LongTensor(input_ids).to(state.device),
                'attention_mask': torch.LongTensor(attention_mask).to(state.device)
                }

    @torch.no_grad()
    def predict(self, state: State, texts):
        result = []
        batches = split_into_batches(texts, state.config.batch_size)
        for batch in batches:
            encoded = self.encode(state, batch)
            pred = self(encoded['input_ids'], encoded['attention_mask'])
            result += [x[0] for x in pred]
        return result

    def save(self, suffix):
        output_dir = Path(".")
        output_dir = os.path.join(
            output_dir, 'roberta-wc-model-{}.bin'.format(suffix))
        torch.save(self.state_dict(), output_dir)
        print("Saved model to {}".format(output_dir))


def train(state, model, dataset, save_to_wandb=False):
    print('start training...')
    np.random.seed(123)
    learning_rate = 3e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataset))
    model.train()
    print('training... num batches:', len(dataset))
    if save_to_wandb:
        init_wandb(name='test-roberta-wc-training-1k')

    criterion = torch.nn.L1Loss()
    for batch in tqdm(dataset):
        encoded = model.encode(state, batch)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        target = list(map(lambda x: [x.relative_position], batch))
        target = torch.tensor(target).to(state.device)

        optimizer.zero_grad()
        pred = model(input_ids, attention_mask)
        loss = criterion(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        if save_to_wandb:
            wandb.log({'roberta_loss': loss.item()})

    if save_to_wandb:
        wandb.finish()
