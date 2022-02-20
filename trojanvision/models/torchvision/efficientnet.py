#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
import torchvision.models
from torchvision.models.efficientnet import model_urls as urls

from collections import Callable


class _EfficientNet(_ImageModel):

    def __init__(self, name: str = 'efficientnet_b0', **kwargs):
        super().__init__(**kwargs)
        ModelClass: Callable[..., torchvision.models.EfficientNet] = getattr(torchvision.models,
                                                                             name.replace('_comp', ''))
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
    available_models = ['efficientnet', 'efficientnet_comp',
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                        'efficientnet_b6', 'efficientnet_b7',
                        'efficientnet_b0_comp', 'efficientnet_b1_comp', 'efficientnet_b2_comp',
                        'efficientnet_b3_comp', 'efficientnet_b4_comp', 'efficientnet_b5_comp',
                        'efficientnet_b6_comp', 'efficientnet_b7_comp']
    model_urls = urls

    def __init__(self, name: str = 'efficientnet',
                 model: type[_EfficientNet] = _EfficientNet, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
