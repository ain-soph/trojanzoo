# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
import torchvision.models as models

"""
This customized network only suitable for MNIST, if implementing on 
CIFAR10 then will get very terrible accuracy (50~60%).

to save clean model:
python train.py -m latentnet --epoch 100 --dataset mnist --save

Epoch: [  50 / 100 ]      Loss: 0.0300,  	 Top1 Acc: 99.083, 	 Top5 Acc: 99.998, 	 Time: 0:00:02
Validate:                 Loss: 0.0214,  	 Top1 Acc: 99.370, 	 Top5 Acc: 100.000, 	 Time: 0:00:00
Model latentnet saved at:  ./data/image/mnist/model/latentnet.pth
"""


class _LatentNet(_ImageModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 16, 5, 1)),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(16, 32, 5, 1)),
        ]))
        self.pool = nn.Sequential(OrderedDict([
            ('maxpool2', nn.MaxPool2d(2, 2)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout2d(0.25))
        ]))


class LatentNet(ImageModel):

    def __init__(self, name='latentnet', model_class=_LatentNet, **kwargs):
        super().__init__(name=name, model_class=model_class,
                         conv_dim=512, fc_depth=2, fc_dim=128, **kwargs)
