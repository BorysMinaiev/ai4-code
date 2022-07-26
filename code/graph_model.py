import torch.nn.functional as F
import os
from pathlib import Path
from urllib.parse import uses_relative
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import wandb
from wandb_helper import init_wandb
from tqdm import tqdm
from dataclasses import dataclass

from common import split_into_batches
from state import State

import torch

from dataclasses import dataclass
import itertools


@dataclass
class Sample:
    markdown: str
    code: str


@dataclass
class SampleWithLabel:
    sample: Sample
    label: float


@dataclass
class TwoSamples:
    markdown: str
    correct_code: str
    wrong_code: str

    def max_len(self):
        l1 = len(self.markdown) + len(self.correct_code)
        l2 = len(self.markdown) + len(self.wrong_code)
        return max(l1, l2)


max_tokenizer_len = 512


class MyGraphModel(nn.Module):
    def __init__(self, state: State, preload_state=None):
        super(MyGraphModel, self).__init__()
        self.graph = AutoModel.from_pretrained(
            'microsoft/graphcodebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/graphcodebert-base')
        self.top = nn.Linear(768, 1)
        if preload_state is not None:
            print('Preloading state:', preload_state)
            self.load_state_dict(torch.load(
                preload_state, map_location=state.device))
        self.name = preload_state if preload_state is not None else "0"

    def forward(self, input_ids, attention_mask, use_sigmoid=True):
        x = self.graph(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = self.top(x[:, 0, :])
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def encode_sample(self, sample):
        max_length = (max_tokenizer_len // 2 - 3)
        code_tokens = self.tokenizer.tokenize(
            sample.code, max_length=max_length, truncation=True)
        markdown_tokens = self.tokenizer.tokenize(
            sample.markdown, max_length=max_length, truncation=True)
        tokens = [self.tokenizer.cls_token] + markdown_tokens + \
            [self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {'input_ids': token_ids, 'attention_mask': [1] * len(token_ids)}

    def encode(self, state: State, samples):
        input_ids = []
        attention_mask = []

        for sample in samples:
            encoded = self.encode_sample(sample)
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])

        max_len = max(map(lambda x: len(x), input_ids))
        for i in range(len(input_ids)):
            more = max_len - len(input_ids[i])
            input_ids[i] += [self.tokenizer.pad_token_id] * more
            attention_mask[i] += [0] * more

        return {'input_ids': torch.LongTensor(input_ids).to(state.device),
                'attention_mask': torch.LongTensor(attention_mask).to(state.device)
                }

    @torch.no_grad()
    def predict(self, state: State, samples):
        result = []
        batches = split_into_batches(samples, state.config.batch_size)
        for batch in batches:
            encoded = self.encode(state, batch)
            pred = self(encoded['input_ids'], encoded['attention_mask'])
            result += [x[0].item() for x in pred]
        return result

    def save(self, suffix, optimizer=None):
        output_dir = Path(".")
        output_path = os.path.join(
            output_dir, 'graph-model-{}.bin'.format(suffix))
        torch.save(self.state_dict(), output_path)
        print("Saved model to {}".format(output_path))
        if optimizer is not None:
            output_path = os.path.join(
                output_dir, 'graph-model-{}.opt.bin'.format(suffix))
            torch.save(optimizer.state_dict(), output_path)


def train(state, model, dataset, save_to_wandb=False, optimizer_state=None):
    print('start training...')
    np.random.seed(123)
    learning_rate = 5e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    if optimizer_state is not None:
        print('loading optimizer state...')
        optimizer.load_state_dict(torch.load(optimizer_state))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05*len(dataset), num_training_steps=len(dataset))
    model.train()
    print('training... num batches:', len(dataset))
    if save_to_wandb:
        init_wandb(name='graph-training')

    criterion = torch.nn.BCELoss()
    for b_id, batch in enumerate(tqdm(dataset)):
        samples = list(map(lambda x: x.sample, batch))
        encoded = model.encode(state, samples)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        target = list(map(lambda x: [x.label], batch))
        target = torch.FloatTensor(target).to(state.device)

        optimizer.zero_grad()
        pred = model(input_ids, attention_mask)

        loss = criterion(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        if save_to_wandb:
            wandb.log({'graph_loss': loss.item()})

        if (b_id % 1000 == 999):
            print('Saving model after', b_id)
            model.save('step-batch-' + str(b_id), optimizer=optimizer)

    if save_to_wandb:
        wandb.finish()

    model.save('cur-final', optimizer=optimizer)


def train2(state, model, dataset, save_to_wandb=False, optimizer_state=None):
    print('start training...')
    np.random.seed(123)
    learning_rate = 5e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    if optimizer_state is not None:
        print('loading optimizer state...')
        optimizer.load_state_dict(torch.load(optimizer_state))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05*len(dataset), num_training_steps=len(dataset))
    model.train()
    print('training... num batches:', len(dataset))
    if save_to_wandb:
        init_wandb(name='graph2-training')

    for b_id, batch in enumerate(tqdm(dataset)):
        samples = list(map(lambda x: [Sample(markdown=x.markdown, code=x.correct_code), Sample(
            markdown=x.markdown, code=x.wrong_code)], batch))
        samples = list(itertools.chain(*samples))
        encoded = model.encode(state, samples)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        optimizer.zero_grad()
        pred = model(input_ids, attention_mask, use_sigmoid=False)

        losses = []
        for i in range(len(batch)):
            score_correct = pred[i * 2 + 0]
            score_wrong = pred[i * 2 + 1]
            cur_scores = torch.tensor(
                [score_wrong, score_correct], requires_grad=True)
            losses.append(F.softmax(cur_scores, dim=0)[0])

        total_loss = sum(losses) / len(batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        if save_to_wandb:
            wandb.log({'graph_loss': total_loss.item()})

        if (b_id % 1000 == 999):
            print('Saving model after', b_id)
            model.save('2-step-batch-' + str(b_id), optimizer=optimizer)

    if save_to_wandb:
        wandb.finish()

    model.save('2-cur-final', optimizer=optimizer)
