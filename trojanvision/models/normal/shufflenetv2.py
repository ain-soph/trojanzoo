#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.shufflenetv2 import model_urls as urls
from collections import OrderedDict
from collections.abc import Callable


class _ShuffleNetV2(_ImageModel):
    def __init__(self, name: str = 'shufflenetv2_x1.0', **kwargs):
        try:
            sub_type: str = name[13:]
            assert sub_type in ['x0.5', 'x1.0', 'x1.5', 'x2.0',
                                'x0.5_comp', 'x1.0_comp', 'x1.5_comp', 'x2.0_comp'], f'{name=}'
        except Exception:
            raise AssertionError("model name should be in ["
                                 "'shufflenetv2_x0.5', 'shufflenetv2_x1.0', "
                                 "'shufflenetv2_x1.5', 'shufflenetv2_x2.0', "
                                 "'shufflenetv2_x0.5_comp', 'shufflenetv2_x1.0_comp', "
                                 "'shufflenetv2_x1.5_comp', 'shufflenetv2_x2.0_comp']")
        super().__init__(**kwargs)
        arch = sub_type[1:4].replace('.', '_')
        ModelClass: Callable[..., torchvision.models.ShuffleNetV2] = getattr(
            torchvision.models, f'shufflenet_v2_x{arch}')
        _model = ModelClass(num_classes=self.num_classes)
        module_list: list[nn.Module] = []
        if 'comp' in sub_type:
            conv1: nn.Conv2d = _model.conv1[0]
            _model.conv1[0] = nn.Conv2d(conv1.in_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                                        stride=1, padding=conv1.padding, dilation=conv1.dilation,
                                        groups=conv1.groups, bias=False)
            module_list.append(('conv1', _model.conv1))
        else:
            module_list.append(('conv1', _model.conv1))
            module_list.append(('maxpool', _model.maxpool))
        module_list.extend([
            ('stage2', _model.stage2),
            ('stage3', _model.stage3),
            ('stage4', _model.stage4),
            ('conv5', _model.conv5),
        ])
        self.features = nn.Sequential(OrderedDict(module_list))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.fc)
        ]))


class ShuffleNetV2(ImageModel):
    available_models = ['shufflenetv2',
                        'shufflenetv2_x0.5', 'shufflenetv2_x1.0', 'shufflenetv2_x1.5', 'shufflenetv2_x2.0',
                        'shufflenetv2_x0.5_comp', 'shufflenetv2_x1.0_comp', 'shufflenetv2_x1.5_comp', 'shufflenetv2_x2.0_comp', ]

    model_urls = urls

    def __init__(self, name: str = 'shufflenetv2', model: type[_ShuffleNetV2] = _ShuffleNetV2, **kwargs):
        super().__init__(name=name, model=model, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        new_dict = OrderedDict()
        for i, (key, value) in enumerate(_dict.items()):
            prefix = 'features.' if i < len(_dict) - 2 else 'classifier.'
            new_dict[prefix + key] = value
        return new_dict
