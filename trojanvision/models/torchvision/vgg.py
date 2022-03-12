#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
import torchvision.models
from torchvision.models.vgg import model_urls as urls

from collections.abc import Callable


class _VGG(_ImageModel):

    def __init__(self, name: str = 'vgg', dropout: float = 0.5, **kwargs):
        if 'num_features' not in kwargs.keys() and ('_comp' or '_s' in name):
            kwargs['num_features'] = [512] if '_s' in name else [512] * 3
        super().__init__(dropout=dropout, **kwargs)
        class_name = name.replace('_comp', '').replace('_s', '')
        ModelClass: Callable[..., torchvision.models.VGG] = getattr(torchvision.models, class_name)
        _model = ModelClass(num_classes=self.num_classes)
        self.features: nn.Sequential = _model.features
        if '_comp' in name:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = _model.avgpool   # nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = _model.classifier
        # nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # ))


class VGG(ImageModel):
    r"""VGG model proposed by Karen Simonyan from University of Oxford in ICLR 2015.

    :Available model names:

        .. code-block:: python3

            ['vgg', 'vgg_bn', 'vgg_comp', 'vgg_bn_comp', 'vgg_s', 'vgg_bn_s',
             'vgg11', 'vgg13', 'vgg16', 'vgg19',
             'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
             'vgg11_comp', 'vgg13_comp', 'vgg16_comp', 'vgg19_comp',
             'vgg11_bn_comp', 'vgg13_bn_comp', 'vgg16_bn_comp', 'vgg19_bn_comp'
             'vgg11_s', 'vgg13_s', 'vgg16_s', 'vgg19_s',
             'vgg11_bn_s', 'vgg13_bn_s', 'vgg16_bn_s', 'vgg19_bn_s']

    See Also:
        * torchvision: :any:`torchvision.models.vgg11`
        * paper: `Very Deep Convolutional Networks for Large-Scale Image Recognition`_

    Note:
        * ``_comp`` sets :any:`torch.nn.AdaptiveAvgPool2d` from ``(7, 7)`` to ``(1, 1)``,
          update the intermediate feature dimension from 4096 to 512 in ``self.classifier``.
        * ``_s`` further makes ``self.classifier`` only one single linear layer based on ``_comp``.

    .. _Very Deep Convolutional Networks for Large-Scale Image Recognition:
        https://arxiv.org/abs/1409.1556
    """
    available_models = ['vgg', 'vgg_bn', 'vgg_comp', 'vgg_bn_comp', 'vgg_s', 'vgg_bn_s',
                        'vgg11', 'vgg13', 'vgg16', 'vgg19',
                        'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                        'vgg11_comp', 'vgg13_comp', 'vgg16_comp', 'vgg19_comp',
                        'vgg11_bn_comp', 'vgg13_bn_comp', 'vgg16_bn_comp', 'vgg19_bn_comp'
                        'vgg11_s', 'vgg13_s', 'vgg16_s', 'vgg19_s',
                        'vgg11_bn_s', 'vgg13_bn_s', 'vgg16_bn_s', 'vgg19_bn_s']

    model_urls = urls

    def __init__(self, name: str = 'vgg', layer: int = 13,
                 model: type[_VGG] = _VGG, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)
