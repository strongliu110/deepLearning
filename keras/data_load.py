#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from glob import glob
import os
import cv2


def get_data_df(basepath):
    img_names = []
    img_labels = []
    img_paths = []
    img_sizes = []
    for path in glob(basepath + "*"):
        label = path.split("/")[-1]
        label_dir = os.path.join(basepath, label)
        for img_name in os.listdir(label_dir):
            img_names.append(img_name)
            img_labels.append(label)
            img_path = os.path.join(label_dir, img_name)
            img_paths.append(img_path)
            # img_sizes.append(cv2.imread(img_path) .shape)
            img_sizes.append([1, 1])

    data_df = pd.DataFrame({'Name': img_names,
                            'Path': img_paths,
                            'Label': img_labels,
                            'Width': list(map(lambda x: x[1], img_sizes)),
                            'Height': list(map(lambda x: x[0], img_sizes))})

    print("data shape: {}".format(data_df.shape))
    data_df = data_df[['Name', 'Path', 'Width', 'Height', 'Label']]
    data_df.sort_values(by=['Name', 'Label'], inplace=True)
    return data_df


def get_test_df(basepath):
    img_names = []
    img_paths = []
    for path in glob(basepath + "*"):
        img_names.append(path.split('/')[-1])
        img_path = os.path.join(basepath, path)
        img_paths.append(img_path)

    data_df = pd.DataFrame({'Name': img_names,
                            'Path': img_paths})
    return data_df


def get_kfold_train_val_idx(df, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
    for train_index, val_index in skf.split(df, df['Label']):
        print("train data size: {}, test data size: {}".format(train_index.shape[0], val_index.shape[0]))
        return train_index, val_index


def get_train_val_df(data_df):
    train_df, val_df, train_label, val_label = train_test_split(data_df, data_df['Label'], test_size=0.1,
                                                                random_state=2, shuffle=True, stratify=data_df['Label'])
    print("train data size: {}, test data size: {}".format(train_df.shape[0], val_df.shape[0]))
    return train_df, val_df, train_label, val_label


def load_train_val_df(data_df, select_idxs, input_shape):
    datas = []
    labels = []
    for idx in select_idxs:
        path = data_df.loc[idx, 'Path']
        image = cv2.resize(cv2.imread(path), (input_shape[0], input_shape[1]))
        label = data_df.loc[idx, 'EncodeLabel']
        image = image / 255  # 归一化
        datas.append(image)  # ravel:返回引用，flatten:返回拷贝
        labels.append(label)
    return np.array(datas), np.array(labels)


def load_test_df(test_df, input_shape):
    datas = []
    for path in test_df['Path']:
        image = cv2.resize(cv2.imread(path), (input_shape[0], input_shape[1]))
        image = image / 255  # 归一化
        datas.append(image)
    return np.array(datas)
