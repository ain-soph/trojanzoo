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

    def get_all_layer(self, x: torch.Tensor, layer_input='input'):
        od = OrderedDict()
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        for block_name, block in self.features.named_children():
            if 'dense' in block_name:
                for layer_name, layer in block.named_children():
                    if record:
                        x = layer(x)
                        od['features.' + block_name + '.' + layer_name] = x
                    if 'features.' + block_name + '.' + layer_name == layer_input:
                        record = True
            elif record:
                x = layer(x)
            od['features.' + layer_name] = x
            if 'features.' + layer_name == layer_input:
                record = True
        if layer_input == 'features':
            record = True
        if record:
            od['features'] = x
            x = self.pool(x)
            od['pool'] = x
            x = x.flatten(start_dim=1)

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                od['classifier.' + name] = x
            elif 'classifier.' + name == layer_input:
                record = True
        od['classifier'] = x
        od['logits'] = x
        od['output'] = x
        return od

    def get_layer_name(self, extra=True):
        layer_name_list = []
        for block_name, block in self.features.named_children():
            if 'dense' in block_name:
                for layer_name, layer in block.named_children():
                    if 'bn' not in block_name and 'relu' not in block_name:
                        layer_name_list.append('features.' + block_name + '.' + layer_name)
            elif 'bn' not in block_name and 'relu' not in block_name:
                layer_name_list.append('features.' + block_name)
        if extra:
            layer_name_list.append('pool')
            layer_name_list.append('flatten')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name_list.append('classifier.' + name)
        return layer_name_list


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
            print('Model {} loaded From Official Website: '.format(self.name), url)
