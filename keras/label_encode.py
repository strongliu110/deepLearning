#!/usr/bin/env python

import numpy as np
import pandas as pd

from sklearn import preprocessing
from keras.utils.np_utils import to_categorical


class Label(object):

    def __init__(self):
        self.le = preprocessing.LabelEncoder()

    def encode(self, train_label):
        self.le.fit(train_label)
        print("Classes: {}".format(self.le.classes_))
        encode_train_labels = self.le.transform(train_label)

        clear_train_label = to_categorical(encode_train_labels)
        num_clases = clear_train_label.shape[1]
        print("Number of classes: {}".format(num_clases))

        return clear_train_label.tolist()

    def decode(self, pred_Y):
        pred_num = np.argmax(pred_Y, axis=1)  # 取最大概率
        pred_label = self.le.classes_[pred_num]
        return pred_label