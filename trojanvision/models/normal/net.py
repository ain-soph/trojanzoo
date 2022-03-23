#!/usr/bin/env python3

# https://github.com/pytorch/examples/blob/main/mnist/main.py

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
from collections import OrderedDict


class _Net(_ImageModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, 1)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('relu2', nn.ReLU(True)),
        ]))
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.25)),
            ('fc1', nn.Linear(9216, 128)),
            ('relu1', nn.ReLU(True)),
            ('dropout2', nn.Dropout(0.5)),
            ('fc2', nn.Linear(128, self.num_classes)),
        ]))


class Net(ImageModel):
    available_models = ['net']

    def __init__(self, name: str = 'net', model: type[_Net] = _Net, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
