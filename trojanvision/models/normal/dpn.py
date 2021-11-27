#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs import dpn

import torch.nn as nn
from collections import OrderedDict
from collections.abc import Callable


class _DPN(_ImageModel):

    def __init__(self, name: str = 'dpn92', **kwargs):
        super().__init__(**kwargs)
        ModelClass: Callable[..., dpn.DPN] = getattr(dpn, name.replace('_comp', ''))
        _model = ModelClass(num_classes=self.num_classes)
        module_list: list[nn.Module] = []
        if 'comp' in name:
            conv1: nn.Conv2d = _model.conv1
            _model.conv1 = nn.Conv2d(conv1.in_channels, conv1.out_channels,
                                     kernel_size=3, stride=1, padding=1, bias=False)
            module_list.extend([
                ('conv1', _model.conv1),
                ('bn1', _model.bn1),
                ('relu', nn.ReLU(inplace=True)),
            ])
        else:
            module_list.extend([
                ('conv1', _model.conv1),
                ('bn1', _model.bn1),
                ('relu', nn.ReLU(inplace=True)),
                ('maxpool', _model.maxpool),
            ])
        module_list.extend([
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4),
        ])
        self.features = nn.Sequential(OrderedDict(module_list))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.fc)
        ]))


class DPN(ImageModel):
    available_models = ['dpn', 'dpn_comp',
                        'dpn68', 'dpn92', 'dpn98', 'dpn131', 'dpn107',
                        'dpn68_comp', 'dpn92_comp', 'dpn98_comp', 'dpn131_comp', 'dpn107_comp']

    def __init__(self, name: str = 'dpn', layer: int = 92,
                 model: type[_DPN] = _DPN, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)
