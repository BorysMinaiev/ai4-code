from unixcoder import UniXcoder
import torch
from config import Config



def reload_model(config:Config, preload=None):
    global unixcoder_model
    global device
    default_model_name = config.unixcoder_model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = None
    if preload is not None:
        print("Preloading from input...")
        state_dict = "/home/ai4-code/code/" + preload 
    unixcoder_model = UniXcoder(model_name=default_model_name, state_dict=state_dict)
    unixcoder_model.to(device)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm
from pathlib import Path

import time

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


def score_answer(nb_id, order):
    ground_truth = [df_orders[nb_id]]
    predictions = [order]

    return kendall_tau(ground_truth, predictions)

def get_random_test_nb_ids(n, random_state=1):
    return df.reset_index(level='cell_id').sample(n=n, random_state=random_state).index.values.tolist()

def calc_nb_score(nb_id):
    return score_answer(nb_id, find_best_cell_order2(df.loc[nb_id]))

def save_results():
    res = []

    for id_, notebook in test_df.groupby(level="id"):
        order = find_best_cell_order2(notebook.reset_index(level='id'))
        order = " ".join(order)
        
        res.append({'id' : id_, 'cell_order' : order})
        
    
    res = pd.DataFrame(res)
    res.to_csv('submission.csv', index=False)

    display(res.head())
    
def get_tokens(text):
    return unixcoder_model.tokenize([text],max_length=512,mode="<encoder-only>")    

def save_model(model, suffix):
    output_dir = Path(".")
    model_to_save = model.encoder.model
    output_dir = os.path.join(output_dir, 'model-{}.bin'.format(suffix)) 
    torch.save(model_to_save.state_dict(), output_dir)
    print("Saved model to {}".format(output_dir))    

def save_roberta_model(model, suffix):
    output_dir = Path(".")
    model_to_save = model
    output_dir = os.path.join(output_dir, 'roberta-model-{}.bin'.format(suffix)) 
    torch.save(model_to_save.state_dict(), output_dir)
    print("Saved model to {}".format(output_dir))    

    
def get_code_cells(nb):
    return nb[nb['cell_type'] == 'code'].index

def get_markdown_cells(nb):
    return nb[nb['cell_type'] == 'markdown'].index        

def split_into_batches(lst, batch_size):
    num_chunks = (len(lst) + batch_size - 1) // batch_size
    return list(np.array_split(lst, num_chunks))