#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanzoo.datasets import Dataset
from trojanvision.environ import env

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import os
import PIL.Image as Image


class ImageSet(Dataset):

    name: str = 'imageset'
    data_type: str = 'image'
    num_classes = 1000
    n_channel: int = 3
    n_dim = (224, 224)

    def __init__(self, norm_par: dict[str, list[float]] = None,
                 default_model: str = 'resnetcomp18', **kwargs):
        super().__init__(default_model=default_model, **kwargs)
        self.norm_par: dict[str, list[float]] = norm_par
        self.param_list['imageset'] = ['n_channel', 'n_dim', 'norm_par']

    @staticmethod
    def get_transform(**kwargs) -> transforms.ToTensor:
        return transforms.ToTensor()

    def get_dataloader(self, mode: str = None, dataset: Dataset = None, batch_size: int = None, shuffle: bool = None,
                       num_workers: int = None, pin_memory=True, drop_last=False, **kwargs) -> torch.utils.data.DataLoader:
        if batch_size is None:
            batch_size = self.test_batch_size if mode == 'test' else self.batch_size
        if shuffle is None:
            shuffle = True if mode == 'train' else False
        num_workers = num_workers if num_workers is not None else self.num_workers
        if dataset is None:
            dataset = self.get_dataset(mode, **kwargs)
        if env['num_gpus'] == 0:
            pin_memory = False
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    @staticmethod
    def get_data(data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return data[0].to(env['device'], non_blocking=True), data[1].to(env['device'], dtype=torch.long, non_blocking=True)

    @classmethod
    def get_class_to_idx(cls, **kwargs) -> dict[str, int]:
        if 'class_to_idx' in cls.__dict__.keys():
            return getattr(cls, 'class_to_idx')
        return {str(i): i for i in range(cls.num_classes)}

    def initialize_folder(self, img_type: str = '.jpg', **kwargs):
        mode_list: list[str] = ['train', 'valid'] if self.valid_set else ['train']
        class_to_idx = self.get_class_to_idx(**kwargs)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        for mode in mode_list:
            dataset: VisionDataset = self.get_org_dataset(mode, transform=None)
            class_counters = [0] * self.num_classes
            for image, target_class in list(dataset):
                image: Image.Image
                target_class: int
                class_name = idx_to_class[target_class]
                _dir = self.folder_path + self.name + f'/{mode}/{class_name}/'
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                image.save(_dir + f'{class_counters[target_class]}{img_type}')
                class_counters[target_class] += 1
