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


@dataclass
class LearnSample:
    text: str
    relative_position: float


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

    def encode(self, state: State, texts):
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            return_token_type_ids=True,
            truncation=True,
            pad_to_max_length=True
        )
        return {'input_ids': torch.tensor(encoded['input_ids']).to(state.device),
                'attention_mask': torch.tensor(encoded['attention_mask']).to(state.device)
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


def train(state, model, dataset):
    print('start training...')
    np.random.seed(123)
    learning_rate = 3e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataset))
    model.train()
    print('training... num batches:', len(dataset))
    init_wandb(name='test-roberta-training-10k')

    criterion = torch.nn.L1Loss()
    for batch in tqdm(dataset):
        texts = list(map(lambda x: x.text, batch))
        encoded = model.encode(state, texts)
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
        wandb.log({'roberta_loss': loss.item()})

    wandb.finish()
