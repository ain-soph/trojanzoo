#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.vgg import model_urls
from collections import OrderedDict


class _VGG(_ImageModel):

    def __init__(self, layer: int = 13, comp: bool = False, **kwargs):
        if comp:
            comp_dict = {'conv_dim': 512, 'fc_depth': 3, 'fc_dim': 512}
            for key, value in comp_dict.items():
                if key not in kwargs.keys():
                    kwargs[key] = value
        super().__init__(**kwargs)
        ModelClass: type[torchvision.models.VGG] = getattr(torchvision.models, f'vgg{layer:d}')
        _model = ModelClass(num_classes=self.num_classes)
        self.features: nn.Sequential = _model.features
        if comp:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = _model.avgpool   # nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = _model.classifier
        # nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )


class VGG(ImageModel):
    def __init__(self, name: str = 'vgg', layer: int = 13,
                 model: type[_VGG] = _VGG, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls[f'vgg{self.layer:d}']
        print('get official model weights from: ', url)
        return model_zoo.load_url(url, **kwargs)

    @classmethod
    def split_model_name(cls, name: str, layer: int = None, width_factor: int = None) -> tuple[str, int, int]:
        bn_flag = True if '_bn' in name else False
        name, layer, width_factor = super().split_model_name(name, layer=layer, width_factor=width_factor)
        if bn_flag:
            name = name.replace('_bn', '') + '_bn'
        return name, layer, width_factor
