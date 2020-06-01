# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
import torchvision.transforms as transforms
from typing import Tuple


class GTSRB(ImageFolder):

    name: str = 'gtsrb'
    n_dim: Tuple[int] = (32, 32)
    num_classes: int = 43
    valid_set: bool = False
    url = {'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
    org_folder_name = {'train': 'GTSRB/Training'}

    def __init__(self, loss_weights=True, **kwargs):
        super().__init__(loss_weights=loss_weights, **kwargs)

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
