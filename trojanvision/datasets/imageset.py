#!/usr/bin/env python3

from trojanzoo.datasets import Dataset
from trojanvision.environ import env

import torch
import torch.utils.data
import torchvision.transforms as transforms
import os

from typing import TYPE_CHECKING
from torchvision.datasets import VisionDataset  # TODO: python 3.10
import PIL.Image as Image
if TYPE_CHECKING:
    pass


class ImageSet(Dataset):

    name: str = 'imageset'
    data_type: str = 'image'
    num_classes = 1000
    data_shape = [3, 224, 224]

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.0], 'std': [1.0], },
                 default_model: str = 'resnetcomp18', **kwargs):
        super().__init__(default_model=default_model, **kwargs)
        self.norm_par: dict[str, list[float]] = norm_par
        self.param_list['imageset'] = ['data_shape', 'norm_par']

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

    def get_class_to_idx(self, **kwargs) -> dict[str, int]:
        if hasattr(self, 'class_to_idx'):
            return getattr(self, 'class_to_idx')
        return {str(i): i for i in range(self.num_classes)}

    def initialize_folder(self, img_type: str = '.png', **kwargs):
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
                _dir = os.path.join(self.folder_path, self.name, mode, class_name)
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                image.save(_dir + f'{class_counters[target_class]}{img_type}')
                class_counters[target_class] += 1
