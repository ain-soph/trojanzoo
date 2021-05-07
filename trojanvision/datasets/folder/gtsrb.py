#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
import torchvision.transforms as transforms

from typing import Union


class GTSRB(ImageFolder):

    name = 'gtsrb'
    data_shape = [3, 32, 32]
    num_classes = 43
    valid_set = False
    url = {'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
    md5 = {'train': '513f3c79a4c5141765e10e952eaa2478'}
    org_folder_name = {'train': 'GTSRB/Training'}

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.3403, 0.3121, 0.3214],
                                                           'std': [0.2724, 0.2608, 0.2669], },
                 loss_weights: bool = True, **kwargs):
        return super().__init__(norm_par=norm_par, loss_weights=loss_weights, **kwargs)

    def get_transform(self, mode: str) -> transforms.Compose:
        if mode != 'train':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()])
        else:
            transform = super().get_transform(mode=mode)
        return transform
