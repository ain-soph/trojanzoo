#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
from collections import OrderedDict


class _Net(_ImageModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('relu2', nn.ReLU()),
        ]))
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.25)),
            ('fc1', self.classifier.fc1),
            ('relu1', self.classifier.relu1),
            ('dropout2', nn.Dropout(0.5)),
            ('fc2', self.classifier.fc2),
        ]))


class Net(ImageModel):
    available_models = ['net']

    def __init__(self, name: str = 'net', model: type[_Net] = _Net, **kwargs):
        super().__init__(name=name, model=model, num_features=[9216, 128], **kwargs)
