#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.vgg import model_urls
from collections import OrderedDict


class _VGG(_ImageModel):

    def __init__(self, layer: int = 13, batch_norm: bool = False, comp: bool = False, **kwargs):
        if comp:
            comp_dict = {'conv_dim': 512, 'fc_depth': 3, 'fc_dim': 512}
            for key, value in comp_dict.items():
                if key not in kwargs.keys():
                    kwargs[key] = value
        super().__init__(**kwargs)
        name = f'vgg{layer:d}' + ('_bn' if batch_norm else '')
        ModelClass: type[torchvision.models.VGG] = getattr(torchvision.models, name)
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
        comp = True if 'comp' in name else False
        batch_norm = True if 'bn' in name else False
        super().__init__(name=name, layer=layer, model=model, comp=comp, batch_norm=batch_norm, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls[self.name]
        print('get official model weights from: ', url)
        return model_zoo.load_url(url, **kwargs)
