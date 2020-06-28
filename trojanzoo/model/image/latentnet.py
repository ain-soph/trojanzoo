# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
import torchvision.models as models


class _LatentNet(_ImageModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, 5, 1)),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(16, 32, 5, 1)),
            ('maxpool2', nn.MaxPool2d(2, 2)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout2d(0.25))
        ]))


class LatentNet(ImageModel):

    def __init__(self, name='latentnet', model_class=_LatentNet, **kwargs):
        super().__init__(name=name, model_class=model_class,
                         conv_dim=32, fc_depth=2, fc_dim=512, **kwargs)

    def get_fm_before_outlayer(self, x):
        """
        Get feature map before output layer.
        """
        x = self._model.features(x)
        x = self._model.pool(x)
        x = self._model.flatten(x)
        x = self._model.classifier.fc1(x)
        return x

    def add_new_last_layer(self):
        """
        replace last fc layer with a clean layer, then return its parameters.
        """
        self._model.classifier.fc2 = nn.Linear(self.fc_dim, self.num_classes)
        # for param in self.classifier['fc2'].parameters():
        #     param.requires_grad = True

        return self._model.classifier.fc2.parameters()
        
    def cpu(self):
        self._model.cpu()
        self.model.cpu()