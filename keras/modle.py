#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard)
import os
import math

from data_load import get_kfold_train_val_idx, load_train_val_df
from data_generator import image_generator


class Model(object):

    def __init__(self, model, save_path=''):
        self.model = model
        self.save_path = save_path

    def __register_callbacks(self):
        # learning rate reduction
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,
                                         factor=0.5, min_lr=0.00001)
        # checkpoints
        model_best_path = os.path.normpath(os.path.join(self.save_path, "models.best_weights.hdf5"))
        model_best = ModelCheckpoint(model_best_path, monitor='val_acc', verbose=1, save_best_only=True)

        weights_path = os.path.normpath(os.path.join(self.save_path, "models.{epoch:02d}-{val_acc:.2f}.hdf5"))
        models = ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=False)

        # early_stop
        early_stop = EarlyStopping(monitor='val_acc', patience=30)

        # history
        logger_path = os.path.normpath(os.path.join(self.save_path, "history.csv"))
        logger = CSVLogger(logger_path)

        # tensorboard
        tensorBoard_path = os.path.normpath(os.path.join(self.save_path, "logs"))
        tensorboard = TensorBoard(write_grads=True, log_dir=tensorBoard_path)

        # all callbacks
        return [lr_reduction, model_best, models, early_stop, logger, tensorboard]

    def __save_history(self, history):
        history_path = os.path.normpath(os.path.join(self.save_path, "history.txt"))
        with open(history_path, 'w') as f:
            f.write("params:" + str(history.params) + "\n")
            f.write("history:" + str(history.history) + "\n")

    @staticmethod
    def __generator_batch_data(data_df, select_idxs, input_shape, batch_size=32):
        np.random.shuffle(select_idxs)  # 重洗
        loop_count = range(int(math.ceil(len(select_idxs) / batch_size)))  # 向上取整
        idx_batches = [select_idxs[range(batch_size * i, min(len(select_idxs), batch_size * (i + 1)))] for i in loop_count]

        while True:
            for idx_batch in idx_batches:
                xx, yy = load_train_val_df(data_df, idx_batch, input_shape)

                yield (xx, yy)

    @staticmethod
    def __generator_multiple_batch_data(generator, data_df, select_idxs, input_shape, batch_size=32):
        np.random.shuffle(select_idxs)  # 重洗
        loop_count = range(int(math.ceil(len(select_idxs) / batch_size)))  # 向上取整
        idx_batches = [select_idxs[range(batch_size * i, min(len(select_idxs), batch_size * (i + 1)))] for i in
                       loop_count]

        while True:
            for idx_batch in idx_batches:
                xx, yy = load_train_val_df(data_df, idx_batch, input_shape)
                gen = generator.flow(xx, batch_size=idx_batch.size)
                xx_gen = gen.next()
                yield (xx_gen, yy)

    def fit(self, data_df, input_shape, batch_size=32, epochs=1):
        # 分割数据集
        train_indexs, val_indexs = get_kfold_train_val_idx(data_df)

        # steps
        steps = len(train_indexs) / batch_size
        val_steps = len(val_indexs) / batch_size

        history = self.model.fit_generator(
            generator=self.__generator_batch_data(data_df, train_indexs, input_shape),
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=self.__generator_batch_data(data_df, val_indexs, input_shape),
            validation_steps=val_steps,
            callbacks=self.__register_callbacks())

        self.__save_history(history)

        return history

    def fit_multiple(self, data_df, input_shape, batch_size=32, epochs=1):
        # 分割数据集
        train_indexs, val_indexs = get_kfold_train_val_idx(data_df)

        # steps
        steps = len(train_indexs) / batch_size
        val_steps = len(val_indexs) / batch_size

        # generator
        train_gen = image_generator()
        val_gen = image_generator()

        history = self.model.fit_generator(
            generator=self.__generator_multiple_batch_data(train_gen, data_df, train_indexs, input_shape, batch_size),
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=self.__generator_multiple_batch_data(val_gen, data_df, val_indexs, input_shape, batch_size),
            validation_steps=val_steps,
            callbacks=self.__register_callbacks())

        self.__save_history(history)

        return history

    def fit_disk(self, input_shape, batch_size=32, epochs=1):
        # generator
        train_gen = image_generator()
        val_gen = image_generator()

        train_generator = train_gen.flow_from_directory(directory='data/train',
                                                        target_size=(input_shape[0], input_shape[1]),
                                                        batch_size=batch_size)
        val_generator = val_gen.flow_from_directory(directory='data/validation',
                                                    target_size=(input_shape[0], input_shape[1]),
                                                    batch_size=batch_size)
        history = self.model.fit_generator(generator=train_generator,
                                           validation_data=val_generator,
                                           epochs=epochs,
                                           callbacks=self.__register_callbacks())

        self.__save_history(history)

        return history

    def load_weights(self, file_path):
        self.model.load_weights(file_path)
        print("load weights success")

    def predict(self, X, batch_size=32):
        pred_y = self.model.predict(X, batch_size=batch_size)
        print("predict success")
        return pred_y

