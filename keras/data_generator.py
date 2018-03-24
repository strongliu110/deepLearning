#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator


def image_generator():
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # 缩放因子
        rotation_range=360,  # 旋转角度
        width_shift_range=0.2,  # 水平偏移
        height_shift_range=0.2,  # 垂直偏移
        shear_range=0.2,  # 倾斜
        zoom_range=0.2,  # 缩放
        horizontal_flip=True,  # 随机水平翻转
        vertical_flip=True,  # 随机垂直翻转
        fill_mode='nearest'
    )

    return datagen

