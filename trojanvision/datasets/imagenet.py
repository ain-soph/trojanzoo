#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imagefolder import ImageFolder
from trojanzoo.utils.param import Module

import torchvision.transforms as transforms
from torchvision.datasets import ImageNet as ImageNet_Official
import numpy as np
import os
import json
from trojanzoo import __file__ as root_file
root_dir = os.path.dirname(root_file)


class ImageNet(ImageFolder):

    name = 'imagenet'
    n_dim = (224, 224)
    url = {
        'train': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'valid': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'test': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
    }

    def __init__(self, norm_par={'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize_folder(self):
        ImageNet_Official(root=self.folder_path, split='train', download=True)
        ImageNet_Official(root=self.folder_path, split='val', download=True)
        os.rename(os.path.join(self.folder_path, 'imagenet', 'val'),
                  os.path.join(self.folder_path, 'imagenet', 'valid'))

    @staticmethod
    def get_transform(mode: str) -> transforms.Compose:
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor()])
            # BiT transform
            # transform = transforms.Compose([
            #     transforms.Resize((480, 480)),
            #     transforms.ToTensor()])
        return transform

    def initialize_npz(self, mode_list: list[str] = ['train', 'valid'],
                       transform: transforms.Compose = transforms.Compose([transforms.Resize((256, 256)),
                                                                           transforms.Lambda(lambda x: np.array(x))]),
                       **kwargs):
        super().initialize_npz(mode_list=mode_list, transform=transform, **kwargs)


class Sample_ImageNet(ImageNet):

    name: str = 'sample_imagenet'
    num_classes = 10
    url = {}
    org_folder_name = {}

    def initialize_folder(self):
        _dict = Module(self.__dict__)
        _dict.__delattr__('folder_path')
        imagenet = ImageNet(**_dict)
        class_dict: dict = {}
        json_path = os.path.normpath(os.path.join(root_dir, 'data', self.name, 'class_dict.json'))
        with open(json_path, 'r', encoding='utf-8') as f:
            class_dict: dict = json.load(f)
        imagenet.sample(child_name=self.name, class_dict=class_dict)
