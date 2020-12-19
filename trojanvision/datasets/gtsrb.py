# -*- coding: utf-8 -*-

from .imagefolder import ImageFolder
import torchvision.transforms as transforms


class GTSRB(ImageFolder):

    name = 'gtsrb'
    n_dim = (32, 32)
    num_classes = 43
    valid_set = False
    url = {'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
    org_folder_name = {'train': 'GTSRB/Training'}

    def __init__(self, loss_weights: bool = True, **kwargs):
        super().__init__(loss_weights=loss_weights, **kwargs)

    @classmethod
    def get_transform(cls, mode) -> transforms.Compose:
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(cls.n_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(cls.n_dim),
                transforms.ToTensor(),
            ])
        return transform
