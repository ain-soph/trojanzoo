# -*- coding: utf-8 -*-
from model.model import *


class LeNet(Model):

    def __init__(self, name='lenet', data_dir='./data/', dataset='mnist', num_classes=10, **kwargs):
        super(LeNet, self).__init__(name=name, data_dir=data_dir, dataset=dataset,
                                    num_classes=num_classes, conv_dim=1024, fc_depth=3, fc_dim=200, **kwargs)

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=(3, 3))),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 32, kernel_size=(3, 3))),
            ('relu2', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv3', nn.Conv2d(32, 64, kernel_size=(3, 3))),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(64, 64, kernel_size=(3, 3))),
            ('relu4', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
