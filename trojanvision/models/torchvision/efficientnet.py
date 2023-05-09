#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
import torchvision.models
from torchvision.models.efficientnet import (EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
                                             EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
                                             EfficientNet_B6_Weights, EfficientNet_B7_Weights,
                                             EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights,
                                             EfficientNet_V2_L_Weights)

from collections.abc import Callable


class _EfficientNet(_ImageModel):

    def __init__(self, name: str = 'efficientnet_b0', **kwargs):
        super().__init__(**kwargs)
        ModelClass: Callable[..., torchvision.models.EfficientNet]
        ModelClass = getattr(torchvision.models, name.replace('_comp', ''))
        _model = ModelClass(num_classes=self.num_classes)
        self.features = _model.features
        self.classifier = _model.classifier
        # nn.Sequential(
        #     nn.Dropout(p=dropout, inplace=True),
        #     nn.Linear(lastconv_output_channels, num_classes),
        # )
        if 'comp' in name:
            conv: nn.Conv2d = self.features[0][0]
            conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                             kernel_size=3, padding=1, bias=False)
            self.features[0][0] = conv


class EfficientNet(ImageModel):
    r"""EfficientNet proposed by Mingxing Tan from Google in ICML 2019.

    :Available model names:

        .. code-block:: python3

            {'efficientnet', 'efficientnet_comp',
             'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
             'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
             'efficientnet_b6', 'efficientnet_b7',
             'efficientnet_b0_comp', 'efficientnet_b1_comp', 'efficientnet_b2_comp',
             'efficientnet_b3_comp', 'efficientnet_b4_comp', 'efficientnet_b5_comp',
             'efficientnet_b6_comp', 'efficientnet_b7_comp'}

    See Also:
        * torchvision: :any:`torchvision.models.efficientnet_b0`
        * paper: `EfficientNet\: Rethinking Model Scaling for Convolutional Neural Networks`_

    Note:
        ``_comp`` reduces the first convolutional layer
        from ``kernel_size=7, stride=2, padding=3``

        to ``kernel_size=3, stride=1, padding=1``.

    .. _EfficientNet\: Rethinking Model Scaling for Convolutional Neural Networks:
        https://arxiv.org/abs/1905.11946
    """
    available_models = {'efficientnet', 'efficientnet_comp',
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                        'efficientnet_b6', 'efficientnet_b7',
                        'efficientnet_b0_comp', 'efficientnet_b1_comp', 'efficientnet_b2_comp',
                        'efficientnet_b3_comp', 'efficientnet_b4_comp', 'efficientnet_b5_comp',
                        'efficientnet_b6_comp', 'efficientnet_b7_comp'}
    weights = {
        'efficientnet_b0': EfficientNet_B0_Weights,
        'efficientnet_b1': EfficientNet_B1_Weights,
        'efficientnet_b2': EfficientNet_B2_Weights,
        'efficientnet_b3': EfficientNet_B3_Weights,
        'efficientnet_b4': EfficientNet_B4_Weights,
        'efficientnet_b5': EfficientNet_B5_Weights,
        'efficientnet_b6': EfficientNet_B6_Weights,
        'efficientnet_b7': EfficientNet_B7_Weights,
        'efficientnet_v2_s': EfficientNet_V2_S_Weights,
        'efficientnet_v2_m': EfficientNet_V2_M_Weights,
        'efficientnet_v2_l': EfficientNet_V2_L_Weights,
    }

    def __init__(self, name: str = 'efficientnet', layer: str = '_b0',
                 model: type[_EfficientNet] = _EfficientNet, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)

    @classmethod
    def get_name(cls, name: str, layer: str = '') -> str:
        layer = layer if '_b' not in name else ''
        return super().get_name(name, layer=layer)
