#!/usr/bin/env python

import numpy as np
import pandas as pd
import os

from data_load import get_data_df, get_test_df, load_test_df
from label_encode import Label
from models import get_compile_model, load_pretrained_models
from model import Model

if __name__ == "__main__":
    # 数据加载
    module_path = os.path.dirname(os.path.abspath(__file__))
    path = module_path.split("/")[:-2]
    data_path = os.path.normpath(os.path.join('/'.join(path), 'seedling/input/train/'))
    data_df = get_data_df(data_path)

    # 标签编码
    label = Label()
    label_list = label.encode(data_df['Label'])
    data_df['EncodeLabel'] = label_list

    input_shape = (70, 70, 3)

    # def read_image(path):
    #     train_img = cv2.resize(cv2.imread(path), (10, 10))
    #     return train_img
    # train_df['Input'] = train_df.apply(lambda row: read_image(row.Path), axis=1)

    # 创建模型
    compile_model = get_compile_model('basic', input_shape, len(label_list[0]), filters=64, kernel=3)
    output_path = os.path.join('/'.join(path), 'seedling/output/')
    model = Model(compile_model, output_path)

    # 加载权值
    # model.load_weights(output_path + "weights.best_01-0.12.hdf5")

    # 训练模型
    model.fit_multiple(data_df, input_shape, batch_size=64, epochs=10)

    # 加载模型
    pretrained_models = load_pretrained_models('./hdf5')
    model = Model(pretrained_models)

    # 测试
    test_path = os.path.join('/'.join(path), 'seedling/input/test/')
    test_df = get_test_df(test_path)

    test_datas = load_test_df(test_df, input_shape)

    # 预测
    pred_y = model.predict(test_datas)
    pred_label = label.decode(pred_y)

    # 保存预测值
    res = {'file': test_df['Name'], 'species': pred_label}
    res = pd.DataFrame(res)
    res.to_csv(output_path + "result.csv", index=False)
