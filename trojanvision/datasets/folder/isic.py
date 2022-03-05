#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
from trojanvision.environ import env

import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm

from trojanvision import __file__ as root_file
root_dir = os.path.dirname(root_file)


class ISIC(ImageFolder):

    name: str = 'isic'

    def initialize_folder(self, **kwargs):
        super().initialize_folder(**kwargs)
        self.split_class('train')
        if self.valid_set:
            self.split_class('valid')

    def split_class(self, mode: str = 'train'):
        csv_path = os.path.normpath(os.path.join(root_dir, 'data', self.name, f'{mode}.csv'))
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
        src_folder = os.path.normpath(os.path.join(self.folder_path, self.name, mode))
        if env['tqdm']:
            labels = tqdm(labels[1:], leave=False)
        for label in labels:
            seq = new_dict[label]
            dst_folder = os.path.join(src_folder, label)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            for img in seq:
                src = os.path.join(src_folder, img + '.jpg')
                dest = os.path.join(dst_folder, img + '.jpg')
                shutil.move(src, dest)


class ISIC2018(ISIC):
    r"""ISIC2018 dataset introduced by Noel Codella in 2018.
    It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * paper: `Skin Lesion Analysis Toward Melanoma Detection 2018\: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)`_
        * website: https://challenge.isic-archive.com/data/

    Attributes:
        name (str): ``'isic2018'``
        num_classes (int): ``7``
        data_shape (list[int]): ``[3, 224, 224]``

    .. _Skin Lesion Analysis Toward Melanoma Detection 2018\: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC):
        https://arxiv.org/abs/1902.03368
    """  # noqa: E501

    name: str = 'isic2018'
    num_classes = 7
    url = {'train': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip',
           'valid': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip'}
    md5 = {'train': '0c281f121070a8d63457caffcdec439a',
           'valid': 'c1fbdd4f5468b0d67c61a1b2def87077'}
    org_folder_name = {'train': 'ISIC2018_Task3_Training_Input',
                       'valid': 'ISIC2018_Task3_Validation_Input'}
