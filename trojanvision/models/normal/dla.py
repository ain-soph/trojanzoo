#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs.dla import DLA as ModelClass

import torch.nn as nn
from collections import OrderedDict


class _DLA(_ImageModel):

    def __init__(self, name: str = 'dla', **kwargs):
        super().__init__(**kwargs)
        simple = True if 'simple' in name else False
        _model = ModelClass(num_classes=self.num_classes, simple=simple)
        if 'comp' in name:
            conv1: nn.Conv2d = _model.features[0][0]
            _model.features[0][0] = nn.Conv2d(conv1.in_channels, conv1.out_channels,
                                              kernel_size=3, stride=1, padding=1, bias=False)
            self.features = nn.Sequential(OrderedDict([
                ('base', _model.features.base),
                ('layer1', _model.features.layer1),
                ('layer2', _model.features.layer2),
                ('layer3', _model.features.layer3),
                ('layer4', _model.features.layer4),
                ('layer5', _model.features.layer5),
                ('layer6', _model.features.layer6),
            ]))
        else:
            self.features = _model.features
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.linear)
        ]))


class DLA(ImageModel):

    def __init__(self, name: str = 'dla', model: type[_DLA] = _DLA, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
