#!/usr/bin/env python

from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121, DenseNet169

from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras import optimizers

from ensemble import MeanEnsemble


def __pretrained_model(arch, input_shape, num_classes, pooling='avg'):
    if arch == 'xception':
        model = Xception(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    elif arch == 'vgg16':
        model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    elif arch == 'vgg19':
        model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    elif arch == 'resnet50':
        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    elif arch == 'inceptionV3':
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    elif arch == 'densenet121':
        return DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    elif arch == 'densenet169':
        return DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
    else:
        raise Exception("Not supported architecture: {}".format(arch))

    # for layer in model.layers:
    #     layer.trainable = False  # 固化

    x = model.output

    if arch == 'densenet169' or arch == 'densenet121':
        x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)

    return model


def __basic_model(input_shape, num_classes, filters=64, kernel=3):
    model = Sequential()

    model.add(Conv2D(filters, (kernel, kernel), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters, (kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters * 2, (kernel, kernel), activation='relu'))
    model.add(Conv2D(filters * 2, (kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters * 4, (kernel, kernel), activation='relu'))
    model.add(Conv2D(filters * 4, (kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))

    return model


def __basic_model_BN(input_shape, num_classes, filters=64, kernel=3):
    model = Sequential()

    model.add(Conv2D(filters, (kernel, kernel), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters, (kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters * 2, (kernel, kernel), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters * 2, (kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(filters * 4, (kernel, kernel), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(filters * 4, (kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=3))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation="softmax"))

    return model


def __get_optimizer(opt='adam'):
    # 数据稀疏时，推荐采用自适应算法：RMSprop，Adam. lr:学习率, epsilon:防止除0错误
    if opt == "sgd":
        optimizer = optimizers.SGD(lr=0.01)  # 用时长，可能困于鞍点
    elif opt == "rmsProp":
        optimizer = optimizers.RMSprop(lr=0.001)  # RNN效果好
    elif opt == "adam":
        optimizer = optimizers.Adam(lr=0.001)
    elif opt == 'adagrad':
        return optimizers.Adagrad(lr=0.01)
    elif opt == 'adadelta':
        return optimizers.Adadelta(lr=1.0)
    else:
        raise Exception("Not supported opt: {}".format(opt))

    print("select opt: {}".format(opt))
    print(optimizer.get_config())

    return optimizer


def get_compile_model(arch, input_shape, num_classes, filters=64, kernel=3, opt="adam"):
    if arch == 'basic':
        model = __basic_model(input_shape, num_classes, filters, kernel)
    elif arch == "basicBN":
        model = __basic_model_BN(input_shape, num_classes, filters, kernel)
    else:
        model = __pretrained_model(arch, input_shape, num_classes)

    print("select arch: {}".format(arch))
    model.summary()

    optimizer = __get_optimizer(opt)

    if num_classes == 2:
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_pretrained_models(model_paths):
    if len(model_paths) <= 1:
        return load_model(model_paths)

    models = []
    for path in model_paths:
        models.append(load_model(path))

    return MeanEnsemble(models)
