# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
from .vggface2 import VGGface2
import torchvision.transforms as transforms


class VGGface2_Sample50(VGGface2):
    """docstring for dataset"""

    def __init__(self, name='vggface2_sample50', num_classes=50, **kwargs):
        super(VGGface2_Sample50, self).__init__(
            name=name, num_classes=num_classes, **kwargs)

        self.output_par(name='VGGface2_Sample50')
    def initialize(self):
        self.sample(child_name=self.name, sample_num=self.num_classes)
