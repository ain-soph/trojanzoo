#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.shufflenetv2 import model_urls as urls
from collections import OrderedDict
from collections.abc import Callable


class _ShuffleNetV2(_ImageModel):
    def __init__(self, name: str = 'shufflenetv2_x1_0', **kwargs):
        try:
            assert name in ShuffleNetV2.available_models, f'{name=}'
        except Exception:
            raise AssertionError(f'model name should be in {ShuffleNetV2.available_models}')
        super().__init__(**kwargs)
        ModelClass: Callable[..., torchvision.models.ShuffleNetV2] = getattr(
            torchvision.models, name[:16])
        _model = ModelClass(num_classes=self.num_classes)
        module_list: list[nn.Module] = []
        if '_comp' in name:
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
    r"""ShuffleNet v2 proposed by Ningning Ma from Megvii in ECCV 2018.

    :Available model names:

        .. code-block:: python3

            ['shufflenetv2', 'shufflenetv2_comp',
             'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
             'shufflenetv2_x1_5', 'shufflenetv2_x2_0',
             'shufflenetv2_x0_5_comp', 'shufflenetv2_x1_0_comp',
             'shufflenetv2_x1_5_comp', 'shufflenetv2_x2_0_comp', ]

    See Also:
        * torchvision: :any:`torchvision.models.shufflenet_v2_x0_5`
        * paper: `ShuffleNet V2\: Practical Guidelines for Efficient CNN Architecture Design`_

    Note:
        ``_comp`` reduces the first convolutional layer
        from ``kernel_size=7, stride=2, padding=3``

        to ``kernel_size=3, stride=1, padding=1``,
        and removes the ``maxpool`` layer before block layers.

    .. _ShuffleNet V2\: Practical Guidelines for Efficient CNN Architecture Design:
        https://arxiv.org/abs/1807.11164
    """
    available_models = ['shufflenetv2', 'shufflenetv2_comp',
                        'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
                        'shufflenetv2_x1_5', 'shufflenetv2_x2_0',
                        'shufflenetv2_x0_5_comp', 'shufflenetv2_x1_0_comp',
                        'shufflenetv2_x1_5_comp', 'shufflenetv2_x2_0_comp', ]

    model_urls = urls

    def __init__(self, name: str = 'shufflenetv2', layer: str = '_x0_5',
                 model: type[_ShuffleNetV2] = _ShuffleNetV2, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)

    @classmethod
    def get_name(cls, name: str, layer: str = '') -> str:
        layer = layer if '_x' not in name else ''
        return super().get_name(name, layer=layer)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        new_dict = OrderedDict()
        for i, (key, value) in enumerate(_dict.items()):
            prefix = 'features.' if i < len(_dict) - 2 else 'classifier.'
            new_dict[prefix + key] = value
        return new_dict
