# -*- coding: utf-8 -*-
from ..image_cnn import Image_CNN
from ...imports.universal import *
from collections import OrderedDict


class LeNet_Simple(Model):
    name = 'lenet'

    def __init__(self, data_dir='./data/', dataset='mnist', num_classes=10, **kwargs):
        super(LeNet_Simple, self).__init__(data_dir=data_dir, dataset=dataset,
                                           num_classes=num_classes, conv_dim=0, fc_depth=0, fc_dim=0, **kwargs)

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5, 1)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(20, 50, 5, 1)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('relu2', nn.ReLU()),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4*4*50, 500)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(500, 10)),
        ]))
