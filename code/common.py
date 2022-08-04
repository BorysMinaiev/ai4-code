from dataclasses import dataclass
import time
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import torch
import math
import re
from config import Config

pd.options.display.width = 180
pd.options.display.max_colwidth = 120


def is_interactive_mode():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive'


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


def save_model(model, suffix):
    output_dir = Path(".")
    model_to_save = model.encoder.model
    output_dir = os.path.join(output_dir, 'model-{}.bin'.format(suffix))
    torch.save(model_to_save.state_dict(), output_dir)
    print("Saved model to {}".format(output_dir))


def save_roberta_model(model, suffix):
    output_dir = Path(".")
    output_dir = os.path.join(
        output_dir, 'roberta-model-{}.bin'.format(suffix))
    torch.save(model.state_dict(), output_dir)
    print("Saved model to {}".format(output_dir))


def get_code_cells(nb):
    return nb[nb['cell_type'] == 'code'].index


def get_markdown_cells(nb):
    return nb[nb['cell_type'] == 'markdown'].index


def split_into_batches(lst, batch_size):
    num_chunks = (len(lst) + batch_size - 1) // batch_size
    return list(np.array_split(lst, num_chunks))


def sim(emb1, emb2):
    return torch.einsum("i,i->", emb1, emb2).detach().numpy()


def get_probs_by_embeddings(embeddings, m_cell_id, code_cell_ids, coef_mul):
    markdown_emb = embeddings[m_cell_id]
    sims = [sim(markdown_emb, embeddings[c]) for c in code_cell_ids]
    max_sim = max(sims)
    sims_probs = list(map(lambda x: math.exp((x-max_sim) * coef_mul), sims))
    sum_probs = sum(sims_probs)
    sims_probs = list(map(lambda x: x/sum_probs, sims_probs))
    return sims_probs


def get_best_pos_by_probs(probs):
    scores = [0.0] * len(probs)
    for i in range(len(probs)):
        for j in range(len(probs)):
            scores[j] += abs(i - j) * probs[i]
    return scores.index(min(scores))


@dataclass
class OneCell:
    score: float
    cell_id: str
    cell_type: str


CLEANR = re.compile('<.*?>')

# TODO: hope it will not timeout...


def clean_html(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext
