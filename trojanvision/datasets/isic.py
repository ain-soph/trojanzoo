#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imagefolder import ImageFolder
from trojanvision.environ import env

import torchvision.transforms as transforms
import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm

from trojanzoo import __file__ as root_file
root_dir = os.path.dirname(root_file)


class ISIC(ImageFolder):

    name: str = 'isic'
    n_dim: tuple = (224, 224)
    valid_set: bool = False

    def initialize_folder(self, **kwargs):
        super().initialize_folder(**kwargs)
        self.split_class()

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
        return transform

    def initialize_npz(self, mode_list: list[str] = ['train', 'valid'],
                       transform: transforms.Compose = transforms.Compose([transforms.Resize(256),
                                                                           transforms.Lambda(lambda x: np.array(x))]),
                       **kwargs):
        super().initialize_npz(mode_list=mode_list, transform=transform, **kwargs)

    def split_class(self):
        csv_path = os.path.normpath(os.path.join(root_dir, 'data', self.name, 'label.csv'))
        print(f'Collect Label Information from CSV file: {csv_path}')
        obj = pd.read_csv(csv_path)
        labels = list(obj.columns.values)
        org_dict = {}
        for label in labels:
            org_dict[label] = np.array(obj[label]) if label == 'image' else np.array([
                bool(value) for value in obj[label]])
        new_dict = {}
        for label in labels[1:]:
            new_dict[label] = org_dict['image'][org_dict[label]]

        print('Splitting dataset to class folders ...')

        src_folder = self.folder_path + self.name + '/train/'
        if env['tqdm']:
            labels = tqdm(labels[1:])
        for label in labels:
            seq = new_dict[label]
            dst_folder = f'{src_folder}{label}/'
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            for img in seq:
                src = src_folder + img + '.jpg'
                dest = dst_folder + img + '.jpg'
                shutil.move(src, dest)


class ISIC2018(ISIC):

    name: str = 'isic2018'
    num_classes = 7
    url = {'train': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip'}
    org_folder_name = {'train': 'ISIC2018_Task3_Training_Input'}
