# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
from package.imports.universal import *
import torchvision.transforms as transforms


class GTSRB(ImageFolder):
    """docstring for dataset"""

    def __init__(self, name='gtsrb', n_dim=(32, 32), num_classes=43, loss_weights=True, **kwargs):
        super(GTSRB, self).__init__(name=name, n_dim=n_dim,
                                    num_classes=num_classes, loss_weights=loss_weights, **kwargs)
        self.url = {
            'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
        self.output_par(name='GTSRB')

    def get_transform(self, mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(self.n_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.n_dim),
                transforms.ToTensor(),
            ])
        return transform
