#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs.pnasnet import PNASNetA, PNASNetB

import torch.nn as nn
from collections import OrderedDict


class _PNASNet(_ImageModel):

    def __init__(self, cell_type: str = 'b', **kwargs):
        super().__init__(**kwargs)
        assert cell_type in ['a', 'b'], cell_type
        ModelClass = PNASNetA if cell_type == 'a' else PNASNetB
        _model = ModelClass(num_classes=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),
            ('relu', nn.ReLU(inplace=True)),
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4),
            ('layer5', _model.layer5)
        ]))
        # self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.linear)
        ]))


class PNASNet(ImageModel):
    available_models = ['pnasnet', 'pnasnet_a', 'pnasnet_b']

    def __init__(self, name: str = 'pnasnet', cell_type: str = 'b',
                 model: type[_PNASNet] = _PNASNet, **kwargs):
        if name == 'pnasnet':
            name += f'_{cell_type}'
        else:
            assert name in ['pnasnet_a', 'pnasnet_b'], name
            cell_type = 'a' if name == 'pnasnet_a' else 'b'
        super().__init__(name=name, cell_type=cell_type, model=model, **kwargs)
