#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.datasets import ImageSet
import trojanvision.utils.model_archs.resnet_ap as resnet_ap

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import model_urls as urls
from torchvision.models.resnet import conv3x3
import argparse
from collections import OrderedDict
from typing import Callable, Union


def GroupNorm(num_groups: int, num_channels: int = None, eps: float = 1e-5, affine: bool = True,
              device=None, dtype=None) -> nn.GroupNorm:
    num_channels = num_groups if num_channels is None else num_channels
    return nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, device=device, dtype=dtype)


class _ResNet(_ImageModel):

    def __init__(self, name: str = 'resnet18', dataset: ImageSet = None, data_shape: list[int] = None,
                 norm_layer: Callable[..., nn.Module] = None, **kwargs):
        super().__init__(**kwargs)
        if data_shape is None:
            assert isinstance(dataset, ImageSet)
            data_shape = dataset.data_shape
        channel = data_shape[0]
        module_list: list[nn.Module] = []
        if 's' in name.split('_'):
            from trojanvision.utils.model_archs.resnet_s import resnet_s
            _model = resnet_s(nclasses=self.num_classes)
            module_list.append(('conv1', _model.conv1))
            module_list.append(('bn1', _model.bn1))
            module_list.append(('relu', nn.ReLU(inplace=True)))
            self.classifier = nn.Sequential(OrderedDict([
                ('fc', _model.fc)
            ]))
        else:
            model_class = name.replace('_ap', '').replace('_comp', '').replace('_s', '')
            module = resnet_ap if 'ap' in name else torchvision.models
            ModelClass = getattr(module, model_class)
            kwargs = {'pool_size': data_shape[1] // 8} if 'ap' in name else {}
            _model: Union[resnet_ap.ResNet_AP, torchvision.models.ResNet] = ModelClass(num_classes=self.num_classes,
                                                                                       norm_layer=norm_layer, **kwargs)
            if 'comp' in name:
                conv1: nn.Conv2d = _model.conv1
                _model.conv1 = conv3x3(channel, conv1.out_channels)
                if 'resnext' in name:
                    _model.fc = nn.Linear(_model.fc.in_features // 2,
                                          _model.fc.out_features,
                                          bias=_model.fc.bias is None)
                module_list.append(('conv1', _model.conv1))
                module_list.append(('bn1', _model.bn1))
                module_list.append(('relu', _model.relu))
            else:
                module_list.append(('conv1', _model.conv1))
                module_list.append(('bn1', _model.bn1))
                module_list.append(('relu', _model.relu))
                module_list.append(('maxpool', _model.maxpool))
            self.pool = _model.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(OrderedDict([
                ('fc', _model.fc)  # nn.Linear(512 * block.expansion, num_classes)
            ]))
            # block.expansion = 1 if BasicBlock and 4 if Bottleneck
            # ResNet 18,34 use BasicBlock, 50 and higher use Bottleneck
        module_list.extend([('layer1', _model.layer1),
                            ('layer2', _model.layer2),
                            ('layer3', _model.layer3)])
        if not ('comp' in name and 'resnext' in name):
            module_list.append(('layer4', _model.layer4))
        self.features = nn.Sequential(OrderedDict(module_list))
        if channel != 3:
            conv: nn.Conv2d = self.features.conv1
            if conv.in_channels != channel:
                keys = ['out_channels', 'kernel_size', 'stride', 'padding']
                kwargs = {key: getattr(conv, key) for key in keys}
                self.features.conv1 = nn.Conv2d(in_channels=channel, bias=False, **kwargs)


class ResNet(ImageModel):
    available_models = ['resnet', 'resnet_comp', 'resnet_s',
                        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                        'resnet18_comp', 'resnet34_comp', 'resnet50_comp', 'resnet101_comp', 'resnet152_comp',
                        'resnet18_s', 'resnet34_s', 'resnet50_s', 'resnet101_s', 'resnet152_s',
                        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
                        'resnext50_32x4d_comp', 'resnext101_32x8d_comp', 'wide_resnet50_2_comp', 'wide_resnet101_2_comp',
                        'resnet18_ap_comp']

    model_urls = urls

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--norm_layer', choices=['bn', 'gn'], default='bn')
        return group

    def __init__(self, name: str = 'resnet', layer: int = 18,
                 model: type[_ResNet] = _ResNet,
                 norm_layer: Union[str, Callable[..., nn.Module]] = 'bn', **kwargs):
        if isinstance(norm_layer, str):
            norm_layer = nn.BatchNorm2d if norm_layer == 'bn' else GroupNorm
        super().__init__(name=name, layer=layer, model=model, norm_layer=norm_layer, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        new_dict = OrderedDict()
        for i, (key, value) in enumerate(_dict.items()):
            prefix = 'features.' if i < len(_dict) - 2 else 'classifier.'
            new_dict[prefix + key] = value
        return new_dict

    @classmethod
    def get_name(cls, name: str, layer: int = None) -> str:
        prefix = ''
        if name.startswith('wide_'):
            prefix = 'wide_'
            name = name[5:]
        return prefix + super().get_name(name, layer=layer)
