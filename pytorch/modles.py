#!/usr/bin/env python

import torchvision.models as models


def __pretrained_model(arch, input_shape, num_classes, pooling='avg'):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'inceptionV3':
        model = models.inception_v3(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    else:
        raise Exception("Not supported architecture: {}".format(arch))