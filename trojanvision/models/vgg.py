#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.vgg import model_urls
from collections import OrderedDict


class _VGG(_ImageModel):

    # layer 13 or 16
    def __init__(self, layer: int = 13, **kwargs):
        super().__init__(**kwargs)
        ModelClass: type[torchvision.models.VGG] = getattr(torchvision.models, 'vgg' + str(layer))
        _model = ModelClass(num_classes=self.num_classes)
        self.features: nn.Sequential = _model.features
        self.pool = _model.avgpool   # nn.AdaptiveAvgPool2d((7, 7))
        if isinstance(self.classifier, nn.Identity):
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

    # layer 13 or 16
    def __init__(self, name: str = 'vgg', layer: int = 13,
                 model_class: type[_VGG] = _VGG, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls['vgg' + str(self.layer)]
        print('get official model weights from: ', url)
        return model_zoo.load_url(url, **kwargs)


class _VGGcomp(_VGG):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


class VGGcomp(VGG):

    def __init__(self, name: str = 'vggcomp', model_class: type[_VGGcomp] = _VGGcomp, **kwargs):
        super().__init__(name=name, model_class=model_class,
                         conv_dim=512, fc_depth=3, fc_dim=512, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        keys_list: list[str] = list(_dict.keys())
        _dict[keys_list[0]] = self._model.features[0].weight
        _dict[keys_list[1]] = self._model.features[0].bias
        return _dict
