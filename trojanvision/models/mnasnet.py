#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel

import torchvision.models
from torch.utils import model_zoo
from torchvision.models.mnasnet import _MODEL_URLS as model_urls
import re

import torch
# import torch.nn as nn
from collections import OrderedDict


class _MNASNet(_ImageModel):

    def __init__(self, mnas_alpha: float, **kwargs):
        super().__init__(**kwargs)
        _model = torchvision.models.MNASNet(mnas_alpha, num_classes=self.num_classes)
        self.features = _model.layers
        self.classifier = _model.classifier
        # conv: nn.Conv2d = self.features[0]
        # self.features[0] = nn.Conv2d(3, conv.out_channels, 3, padding=1, stride=1, bias=False)


class MNASNet(ImageModel):

    def __init__(self, name: str = 'mnasnet', mnas_alpha: float = 1.0,
                 model: type[_MNASNet] = _MNASNet, **kwargs):
        name, self.mnas_alpha = self.parse_name(name, mnas_alpha)
        super().__init__(name=name, mnas_alpha=self.mnas_alpha, model=model, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls[self.parse_name('mnasnet', self.mnas_alpha)[0]]
        print('get official model weights from: ', url)
        _dict: OrderedDict[str, torch.Tensor] = model_zoo.load_url(url, **kwargs)
        new_dict = OrderedDict()
        for key, value in _dict.items():
            if key.startswith('layers.'):
                key = 'features.' + key[7:]
            new_dict[key] = value
        return new_dict

    @staticmethod
    def parse_name(name: str, mnas_alpha: float = 1.0) -> tuple[str, float]:
        name_list: list[str] = re.findall('[a-zA-Z]+|[\d_.]+', name)
        name = name_list[0]
        if len(name_list) > 1:
            assert len(name_list) == 2
            mnas_alpha = float(name_list[1].replace('_', '.'))
        return f'{name}{mnas_alpha:.1f}'.replace('.', '_'), mnas_alpha

    @classmethod
    def split_model_name(cls, name: str, layer: int = None, width_factor: int = None) -> tuple[str, int, int]:
        return name, layer, width_factor
