#!/usr/bin/env python3

# https://github.com/google-research/big_transfer

from trojanvision.utils.model_archs.bit import KNOWN_MODELS, tf2th, StdConv2d, conv3x3
from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.datasets import ImageNet

import torch
import torch.nn as nn
import torch.hub
import numpy as np
import os
from collections import OrderedDict


class _BiT(_ImageModel):
    def __init__(self, name: str = 'bit-m-r50x1', **kwargs):
        super().__init__(**kwargs)
        model_name = name.split('_')[0].upper().replace('BIT', 'BiT').replace('X', 'x')
        _model = KNOWN_MODELS[model_name](head_size=1)
        self.features = nn.Sequential()
        if 'comp' in name:
            assert 'official' not in name, name
            conv: StdConv2d = _model.root.conv
            self.features.add_module('conv', conv3x3(conv.in_channels, conv.out_channels))
        else:
            self.features.add_module('conv', _model.root.conv)
            if 'official' not in name:
                pool: nn.MaxPool2d = _model.root.pool
                _model.root.pool = nn.MaxPool2d(pool.kernel_size, pool.stride, padding=1)
            else:
                self.features.add_module('pad', _model.root.pad)
            self.features.add_module('pool', _model.root.pool)
        for name, child in _model.body.named_children():
            self.features.add_module(name, child)
        self.features.add_module('gn', _model.head.gn)
        self.features.add_module('relu', _model.head.relu)
        fc: nn.Conv2d = _model.head.conv
        self.classifier = self.define_classifier(conv_dim=fc.in_channels,
                                                 num_classes=self.num_classes, fc_depth=1)


class BiT(ImageModel):
    available_models = ['bit', 'bit_comp', 'bit_official',
                        'bit-m-r50x1', 'bit-m-r50x3', 'bit-m-r101x1', 'bit-m-r101x3', 'bit-m-r152x2', 'bit-m-r152x4',
                        'bit-s-r50x1', 'bit-s-r50x3', 'bit-s-r101x1', 'bit-s-r101x3', 'bit-s-r152x2', 'bit-s-r152x4',
                        'bit-m-r50x1_comp', 'bit-m-r50x3_comp', 'bit-m-r101x1_comp',
                        'bit-m-r101x3_comp', 'bit-m-r152x2_comp', 'bit-m-r152x4_comp',
                        'bit-s-r50x1_comp', 'bit-s-r50x3_comp', 'bit-s-r101x1_comp',
                        'bit-s-r101x3_comp', 'bit-s-r152x2_comp', 'bit-s-r152x4_comp',
                        'bit-m-r50x1_official', 'bit-m-r50x3_official', 'bit-m-r101x1_official',
                        'bit-m-r101x3_official', 'bit-m-r152x2_official', 'bit-m-r152x4_official',
                        'bit-s-r50x1_official', 'bit-s-r50x3_official', 'bit-s-r101x1_official',
                        'bit-s-r101x3_official', 'bit-s-r152x2_official', 'bit-s-r152x4_official']

    def __init__(self, name: str = 'bit',
                 pretrained_dataset: str = 'm', layer: int = 50, width_factor: int = 1,
                 model: type[_BiT] = _BiT, norm_par: dict[str, list[float]] = None,
                 official: bool = False, **kwargs):
        name = self.parse_name(name, pretrained_dataset, layer, width_factor)
        if official and norm_par is None:
            norm_par = {'mean': [0.5, 0.5, 0.5],
                        'std': [0.5, 0.5, 0.5], }
        super().__init__(name=name, width_factor=width_factor,
                         model=model, norm_par=norm_par,
                         official=official, **kwargs)

    @staticmethod
    def parse_name(name: str, pretrained_dataset: str = 'm', layer: int = 50, width_factor: int = 1) -> str:
        full_name_list = name.split('_')
        name_list = full_name_list[0].lower().split('-')
        assert name_list[0] == 'bit', name
        if len(name_list) != 1:
            for element in name_list[1:]:
                if element[0] == 'r':
                    sub_list = element[1:].split('x')
                    layer = int(sub_list[0])
                    if len(sub_list) == 2:
                        width_factor = int(sub_list[1])
                else:
                    assert len(element) == 1, name
                    pretrained_dataset = element
        full_name_list[0] = '-'.join(['bit', pretrained_dataset, f'r{layer:d}x{width_factor:d}'])
        return '_'.join(full_name_list)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        # TODO: map_location argument
        assert 'official' in self.name and 'comp' not in self.name, self.name
        model_name = self.name.split('_')[0].upper().replace('BIT', 'BiT').replace('X', 'x')
        if isinstance(self.dataset, ImageNet):
            model_name += '-ILSVRC2012'
        url = f'https://storage.googleapis.com/bit_models/{model_name}.npz'
        print('get official model weights from: ', url)
        file_path = os.path.join(torch.hub.get_dir(), 'bit', f'{model_name}.npz')
        if not os.path.exists(file_path):
            torch.hub.download_url_to_file(url, file_path)
        weights: dict[str, np.ndarray] = np.load(file_path)
        _dict = OrderedDict()
        _dict['features.conv.weight'] = tf2th(weights['resnet/root_block/standardized_conv2d/kernel'])
        for block_num in range(4):
            block_name = f'block{block_num+1:d}'
            block: nn.Sequential = getattr(self._model.features, block_name)
            for unit_name, unit in block.named_children():
                prefix = '/'.join(['resnet', block_name, unit_name, ''])
                dict_prefix = '.'.join(['features', block_name, unit_name, ''])
                _dict[dict_prefix + 'gn1.weight'] = tf2th(weights[prefix + 'a/group_norm/gamma'])
                _dict[dict_prefix + 'gn1.bias'] = tf2th(weights[prefix + 'a/group_norm/beta'])
                _dict[dict_prefix + 'conv1.weight'] = tf2th(weights[prefix + 'a/standardized_conv2d/kernel'])
                _dict[dict_prefix + 'gn2.weight'] = tf2th(weights[prefix + 'b/group_norm/gamma'])
                _dict[dict_prefix + 'gn2.bias'] = tf2th(weights[prefix + 'b/group_norm/beta'])
                _dict[dict_prefix + 'conv2.weight'] = tf2th(weights[prefix + 'b/standardized_conv2d/kernel'])
                _dict[dict_prefix + 'gn3.weight'] = tf2th(weights[prefix + 'c/group_norm/gamma'])
                _dict[dict_prefix + 'gn3.bias'] = tf2th(weights[prefix + 'c/group_norm/beta'])
                _dict[dict_prefix + 'conv3.weight'] = tf2th(weights[prefix + 'c/standardized_conv2d/kernel'])
                if hasattr(unit, 'downsample'):
                    weight = tf2th(weights[prefix + 'a/proj/standardized_conv2d/kernel'])
                    _dict[dict_prefix + 'downsample.weight'] = weight
        _dict['features.gn.weight'] = tf2th(weights['resnet/group_norm/gamma'])
        _dict['features.gn.bias'] = tf2th(weights['resnet/group_norm/beta'])
        _dict['classifier.fc.weight'] = tf2th(weights['resnet/head/conv2d/kernel']).flatten(1)
        _dict['classifier.fc.bias'] = tf2th(weights['resnet/head/conv2d/bias'])
        return _dict
