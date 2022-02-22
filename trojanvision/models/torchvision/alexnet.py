#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
import torchvision.models
from torchvision.models.alexnet import model_urls as urls


class _AlexNet(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _model = torchvision.models.alexnet(num_classes=self.num_classes)
        self.features = _model.features
        self.pool = _model.avgpool   # nn.AdaptiveAvgPool2d((6, 6))
        if len(self.classifier) == 1 and \
                isinstance(self.classifier[0], nn.Identity):
            self.classifier = _model.classifier


class AlexNet(ImageModel):
    r"""AlexNet proposed by Alex Krizhevsky from Google in 2014.

    :Available model names:

        .. code-block:: python3

            ['alexnet']

    See Also:
        * torchvision: :any:`torchvision.models.alexnet`
        * paper: `One weird trick for parallelizing convolutional neural networks`_

    .. code-block:: python3

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten(1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    .. _One weird trick for parallelizing convolutional neural networks:
        https://arxiv.org/abs/1404.5997
    """
    available_models = ['alexnet']
    model_urls = urls

    def __init__(self, name: str = 'alexnet',
                 model: type[_AlexNet] = _AlexNet, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
