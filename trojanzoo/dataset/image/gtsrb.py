# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
import torchvision.transforms as transforms


class GTSRB(ImageFolder):
    name = 'gtsrb'
    n_dim = (32, 32)
    num_classes = 43

    def __init__(self, loss_weights=True, **kwargs):
        self.url = {
            'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
        super(GTSRB, self).__init__(loss_weights=loss_weights, **kwargs)

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
