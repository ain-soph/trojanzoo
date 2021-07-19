#!/usr/bin/env python3

import torch.nn as nn
from trojanvision.datasets import ImageSet
from trojanvision.models.imagemodel import ImageModel, _ImageModel
from collections import OrderedDict


class _ConvNet(_ImageModel):

    def __init__(self, channel=1, im_size=(28, 28), **kwargs):
        super().__init__(**kwargs)

        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        self.net_act = net_act
        self.net_pooling = net_pooling

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.pool = nn.Identity()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_feat, self.num_classes))
        ]))

    def _get_actlayer(self, net_act: str) -> nn.Module:
        if net_act == 'sigmoid':
            act = nn.Sigmoid()
        elif net_act == 'relu':
            act = nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            act = nn.LeakyReLU(negative_slope=0.01)
        else:
            act = None
            exit('unknown activation function: %s' % net_act)
        return act

    def _get_poolinglayer(self, net_pooling: str) -> nn.Module:
        if net_pooling == 'maxpooling':
            pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            pooling = None
        else:
            pooling = None
            exit('unknown net_pooling: %s' % net_pooling)
        return pooling

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s' % net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_actlayer(self.net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_poolinglayer(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat


class ConvNet(ImageModel):
    available_models = ['convnet']

    def __init__(self, name: str = 'convnet', model: type[_ConvNet] = _ConvNet, dataset: ImageSet = None, **kwargs):
        if dataset is not None:
            kwargs['channel'] = dataset.data_shape[0]
            kwargs['im_size'] = dataset.data_shape[1:]
        super().__init__(name=name, model=model, dataset=dataset, **kwargs)
