# -*- coding: utf-8 -*-
from .imagenet import ImageNet
from package.imports.universal import *
import torchvision.transforms as transforms


class Sample_ImageNet(ImageNet):
    """docstring for dataset"""

    def __init__(self, name='sample_imagenet', batch_size=128, num_classes=20, default_model='resnet18', **kwargs):
        super(Sample_ImageNet, self).__init__(name=name, batch_size=batch_size,
                                              num_classes=num_classes, default_model=default_model,  **kwargs)

        self.output_par(name='Sample_ImageNet')

    def initialize(self):
        pass
