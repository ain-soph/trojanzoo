# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

from collections import OrderedDict

import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.alexnet import model_urls
import torchvision.models as models


class _AlexNet(_ImageModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _model: models.AlexNet = models.alexnet(num_classes=self.num_classes)
        self.features = _model.features
        self.pool = _model.avgpool   # nn.AdaptiveAvgPool2d((6, 6))
        if isinstance(self.classifier, nn.Identity):
            self.classifier = _model.classifier

        # nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )

        # nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )


class AlexNet(ImageModel):

    def __init__(self, name='alexnet', model_class=_AlexNet, **kwargs):
        super().__init__(name=name, model_class=model_class, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['alexnet']
        _dict = model_zoo.load_url(url)
        if self.num_classes == 1000:
            self._model.load_state_dict(_dict)
        else:
            new_dict = OrderedDict()
            for name, param in _dict.items():
                if 'classifier.6' not in name:
                    new_dict[name] = param
            self._model.load_state_dict(new_dict, strict=False)
        if verbose:
            print(f'Model {self.name} loaded From Official Website: {url}')
