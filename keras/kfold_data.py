#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_kfold_idx(df, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
    fold_lst = []
    for train_index, test_index in skf.split(df, df['Label']):
        fold_lst.append(test_index)
    return fold_lst


def get_kfold_df(data_df, k=10):
    data_df['Fold'] = k
    kfold_idx = get_kfold_idx(data_df, k)
    for i, idx in enumerate(kfold_idx):
        data_df.loc[idx, 'Fold'] = i
    return data_df
