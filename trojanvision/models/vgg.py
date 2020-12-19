# -*- coding: utf-8 -*-
from .imagemodel import _ImageModel, ImageModel

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
    def __init__(self, name='vgg', layer=None, model_class=_VGG, default_layer=13, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['vgg' + str(self.layer)]
        _dict = model_zoo.load_url(url)
        if self.num_classes == 1000:
            self._model.load_state_dict(_dict)
        else:
            new_dict = OrderedDict()
            for name, param in _dict.items():
                if 'classifier.6' not in name:
                    new_dict[name] = param
            self._model.load_state_dict(new_dict, strict=False)
        if verbose:
            print(f'Model {self.name} loaded From Official Website: {url}')


class _VGGcomp(_VGG):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


class VGGcomp(VGG):

    def __init__(self, name='vggcomp', model_class=_VGGcomp, **kwargs):
        super().__init__(name=name, model_class=model_class,
                         conv_dim=512, fc_depth=3, fc_dim=512, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['vgg' + str(self.layer)]
        _dict = model_zoo.load_url(url)
        new_dict = OrderedDict()
        for name, param in _dict.items():
            if 'classifier' not in name:
                new_dict[name] = param
        self._model.load_state_dict(new_dict, strict=False)
        if verbose:
            print(f'Model {self.name} loaded From Official Website: {url}')
