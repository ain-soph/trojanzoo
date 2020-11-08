# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

import re
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.densenet import model_urls
import torchvision.models as models


class _DenseNet(_ImageModel):

    def __init__(self, layer=121, **kwargs):
        super().__init__(**kwargs)
        _model: models.DenseNet = models.__dict__[
            'densenet' + str(layer)](num_classes=self.num_classes)
        self.features = _model.features
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.classifier)  # nn.Linear(512 * block.expansion, num_classes)
        ]))


class DenseNet(ImageModel):

    def __init__(self, name='densenet', layer=None, model_class=_DenseNet, default_layer=121, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['densenet' + str(self.layer)]
        _dict = model_zoo.load_url(url)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                _dict[new_key] = _dict[key]
                del _dict[key]
        _dict['classifier.fc.weight'] = _dict['classifier.weight']
        _dict['classifier.fc.bias'] = _dict['classifier.bias']
        del _dict['classifier.weight']
        del _dict['classifier.bias']
        if self.num_classes == 1000:
            self._model.load_state_dict(_dict)
        else:
            new_dict = OrderedDict()
            for name, param in _dict.items():
                if 'classifier' not in name:
                    new_dict[name] = param
            self._model.load_state_dict(new_dict, strict=False)
        if verbose:
            print(f'Model {self.name} loaded From Official Website: {url}')


class _DenseNetcomp(_DenseNet):

    def __init__(self, layer=121, **kwargs):
        super().__init__(**kwargs)
        conv = self.features.conv0
        self.features.conv0 = nn.Conv2d(3, conv.out_channels, kernel_size=3, padding=1, bias=False)


class DenseNetcomp(DenseNet):

    def __init__(self, name='densenetcomp', layer=None, model_class=_DenseNetcomp, default_layer=121, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)
