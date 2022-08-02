import math
from common import sim
from cosine_train import end_token
import torch.nn.functional as F
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import wandb
from wandb_helper import init_wandb
from tqdm import tqdm
from dataclasses import dataclass

from common import split_into_batches
from state import State

import torch

from dataclasses import dataclass
import itertools

from torch.optim.lr_scheduler import CosineAnnealingLR


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


max_tokenizer_len = 256


class MyGraphModel(nn.Module):
    def __init__(self, state: State, preload_state=None, next_code_cells=1):
        super(MyGraphModel, self).__init__()
        self.graph = AutoModel.from_pretrained(
            'microsoft/graphcodebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/graphcodebert-base')
        self.top = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.2)
        self.next_code_cells = next_code_cells
        if preload_state is not None:
            print('Preloading state:', preload_state)
            self.load_state_dict(torch.load(
                preload_state, map_location=state.device))
        self.name = preload_state if preload_state is not None else "0"
        self.name += ";ncs=" + str(next_code_cells)

    def forward(self, input_ids, attention_mask, use_sigmoid, return_scalar):
        x = self.graph(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        if return_scalar:
            x = self.top(x)
        else:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
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

    def encode_texts(self, state: State, texts, max_length=510):
        input_ids = []
        attention_mask = []

        for text in texts:
            tokens = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                text, max_length=max_length, truncation=True)) + [self.tokenizer.sep_token_id]
            input_ids.append(tokens)
            attention_mask.append([1] * len(tokens))

        max_len = max(map(lambda x: len(x), input_ids))
        for i in range(len(input_ids)):
            more = max_len - len(input_ids[i])
            input_ids[i] += [self.tokenizer.pad_token_id] * more
            attention_mask[i] += [0] * more

        return {'input_ids': torch.LongTensor(input_ids).to(state.device),
                'attention_mask': torch.LongTensor(attention_mask).to(state.device)
                }

    @torch.no_grad()
    def predict(self, state: State, samples, use_sigmoid):
        result = []
        batches = split_into_batches(samples, state.config.batch_size)
        for batch in batches:
            encoded = self.encode(state, batch)
            pred = self(encoded['input_ids'],
                        encoded['attention_mask'], use_sigmoid)
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
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataset))
    # scheduler = get_linear_schedule_with_warmup(
    #    optimizer, num_warmup_steps=0.05*len(dataset), num_training_steps=len(dataset))
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
    learning_rate = 3e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    #optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.995))
    if optimizer_state is not None:
        print('loading optimizer state...')
        optimizer.load_state_dict(torch.load(optimizer_state))
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0.05*len(dataset), num_training_steps=len(dataset))
    model.train()
    print('training... num batches:', len(dataset))
    if save_to_wandb:
        init_wandb(name='graph2-training')

    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 1
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(dataset)/accumulation_steps)

    for b_id, batch in enumerate(tqdm(dataset)):
        samples = list(map(lambda x: [Sample(markdown=x.markdown, code=x.correct_code), Sample(
            markdown=x.markdown, code=x.wrong_code)], batch))
        samples = list(itertools.chain(*samples))
        encoded = model.encode(state, samples)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        with torch.cuda.amp.autocast():
            pred = model(input_ids, attention_mask, use_sigmoid=False)

            losses = []
            for i in range(len(batch)):
                sm = F.softmax(pred[i*2:i*2+2] * 10.0, dim=0)
                losses.append(sm[1])

            total_loss = sum(losses) / len(batch)
        scaler.scale(total_loss).backward()
        if b_id % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        #

        if save_to_wandb:
            wandb.log({'graph2_loss': total_loss.item()})

        if (b_id % 5000 == 4999):
            print('Saving model after', b_id)
            model.save('2-step-batch-' + str(b_id), optimizer=optimizer)

    if save_to_wandb:
        wandb.finish()

    model.save('2-cur-final', optimizer=optimizer)


@dataclass
class Embedding:
    cell_id: str
    text: str


def get_nb_embeddings(state: State, model, nb):
    to_convert = [Embedding(cell_id=end_token, text=end_token)]
    for cell_id in nb.index:
        to_convert.append(
            Embedding(cell_id=cell_id, text=nb.loc[cell_id]['source']))
    to_convert.sort(key=lambda x: len(x.text))

    num_chunks = (len(to_convert) + state.config.batch_size -
                  1) // state.config.batch_size

    result = {}
    for batch in np.array_split(to_convert, num_chunks):
        all_texts = list(map(lambda x: x.text, batch))
        encoded = model.encode_texts(state, all_texts)
        embeddings = model(
            input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], use_sigmoid=False, return_scalar=False)
        for i in range(len(batch)):
            result[batch[i].cell_id] = embeddings[i].cpu()
    return result
