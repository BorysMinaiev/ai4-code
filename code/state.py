from dataclasses import dataclass
from config import Config
import pandas as pd
from tqdm import tqdm
import common
import torch


@dataclass
class State:
    # TODO: wrong types. How does it work? :)
    df_orders: list
    test_df: list
    df_ancestors: list
    all_train_nb: list
    all_validate_nb: list
    cur_train_nbs: list
    config: Config
    device: str

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.load_df_orders()
        self.load_df_ancestors()

    def load_df_orders(self):
        self.df_orders = pd.read_csv(
            self.config.data_dir / 'train_orders.csv',
            index_col='id',
        ).squeeze("columns").str.split()  # Split the string representation of cell_ids into a list

    def load_test_nbs(self):
        paths_test = list((self.config.data_dir / 'test').glob('*.json'))
        notebooks_test = [
            common.read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
        ]
        self.test_df = (
            pd.concat(notebooks_test)
            .set_index('id', append=True)
            .swaplevel()
            .sort_index(level='id', sort_remaining=False)
        )

    def load_df_ancestors(self):
        self.df_ancestors = pd.read_csv(
            self.config.data_dir / 'train_ancestors.csv', index_col='id')

        # TODO: rewrite this to use the dataframe
        cnt_by_group = {}
        for id, row in tqdm(self.df_ancestors.iterrows()):
            cnt_by_group[row['ancestor_id']] = cnt_by_group.get(
                row['ancestor_id'], 0) + 1

        cnt = pd.Series(cnt_by_group)
        print('only one:', cnt[cnt == 1].count())
        cnt.plot.hist(grid=True, bins=20, rwidth=0.9,
                      color='#607c8e')
        cnt

        good_notebooks = []
        for id, row in tqdm(self.df_ancestors.iterrows()):
            if row['parent_id'] != None and cnt_by_group[row['ancestor_id']] == 1:
                good_notebooks.append(id)

        good_notebooks = pd.Series(good_notebooks)
        print('good notebooks', len(good_notebooks))

        self.all_train_nb = good_notebooks.sample(
            frac=0.9, random_state=787788)
        self.all_validate_nb = good_notebooks.drop(self.all_train_nb.index)

    def load_train_nbs_helper(self, ids):
        paths_train = [self.config.data_dir / 'train' /
                       '{}.json'.format(id) for id in ids]
        notebooks_train = [
            common.read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
        ]
        self.cur_train_nbs = pd.concat(notebooks_train).set_index(
            'id', append=True).swaplevel().sort_index(level='id', sort_remaining=False)

    def load_train_nbs(self, num: int):
        self.load_train_nbs_helper(self.all_train_nb.head(num))

    def load_train_nbs_range(self, from_: int, to_: int):
        self.load_train_nbs_helper(self.all_train_nb[from_:to_])

    def load_train_nbs_tail(self, num: int):
        self.load_train_nbs_helper(self.all_train_nb.tail(num))

    def load_one_nb(self, nb_id):
        self.load_train_nbs_helper([nb_id])
        return self.cur_train_nbs.loc[nb_id]
