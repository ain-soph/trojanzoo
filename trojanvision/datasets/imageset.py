#!/usr/bin/env python3

from trojanzoo.datasets import Dataset
from trojanvision.environ import env
from trojanvision.utils.data import Cutout

import torch
import torchvision.transforms as transforms
import argparse
import os

from typing import TYPE_CHECKING
from torchvision.datasets import VisionDataset  # TODO: python 3.10
import PIL.Image as Image
from typing import Union
if TYPE_CHECKING:
    import torch.utils.data


class ImageSet(Dataset):

    name: str = 'imageset'
    data_type: str = 'image'
    num_classes = 1000
    data_shape = [3, 224, 224]

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--dataset_normalize', dest='normalize',
                           action='store_true', help='use transforms.Normalize in dataset transform. '
                           '(It\'s used in model as the first layer by default.)')
        group.add_argument('--transform', choices=[None, 'bit', 'pytorch'])
        group.add_argument('--auto_augment', action='store_true', help='use auto augment')
        group.add_argument('--cutout', action='store_true', help='use cutout')
        group.add_argument('--cutout_length', type=int, help='cutout length')
        return group

    def __init__(self, norm_par: dict[str, list[float]] = None,
                 default_model: str = 'resnet18_comp',
                 normalize: bool = False, transform: str = None, auto_augment: bool = False,
                 cutout: bool = False, cutout_length: int = None, **kwargs):
        self.norm_par: dict[str, list[float]] = norm_par
        self.normalize = normalize
        self.transform = transform
        self.auto_augment = auto_augment
        self.cutout = cutout
        self.cutout_length = cutout_length
        super().__init__(default_model=default_model, **kwargs)
        self.param_list['imageset'] = ['data_shape', 'norm_par', 'normalize', 'transform', 'auto_augment', 'cutout']
        if cutout:
            self.param_list['imageset'].append('cutout_length')

    def get_transform(self, mode: str, normalize: bool = None) -> transforms.Compose:
        normalize = normalize if normalize is not None else self.normalize
        if self.transform == 'bit':
            return get_transform_bit(mode, self.data_shape)
        elif self.data_shape == [3, 224, 224]:
            transform = get_transform_imagenet(mode, use_tuple=self.transform != 'pytorch',
                                               auto_augment=self.auto_augment)
        elif self.data_shape in ([3, 16, 16], [3, 32, 32]):
            transform = get_transform_cifar(mode, auto_augment=self.auto_augment,
                                            cutout=self.cutout, cutout_length=self.cutout_length,
                                            data_shape=self.data_shape)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        if normalize and self.norm_par is not None:
            transform.transforms.append(transforms.Normalize(mean=self.norm_par['mean'], std=self.norm_par['std']))
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

    def make_folder(self, img_type: str = '.png', **kwargs):
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
                image.save(os.path.join(_dir, f'{class_counters[target_class]}{img_type}'))
                class_counters[target_class] += 1


def get_transform_bit(mode: str, data_shape: list[int]) -> transforms.Compose:
    hyperrule = data_shape[-2] * data_shape[-1] < 96 * 96
    precrop, crop = (160, 128) if hyperrule else (512, 480)
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((precrop, precrop)),
            transforms.RandomCrop((crop, crop)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor()])
    return transform


def get_transform_imagenet(mode: str, use_tuple: bool = False, auto_augment: bool = False) -> transforms.Compose:
    if mode == 'train':
        transform_list = [
            transforms.RandomResizedCrop((224, 224) if use_tuple else 224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ]
        if auto_augment:
            transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256) if use_tuple else 256),
            transforms.CenterCrop((224, 224) if use_tuple else 224),
            transforms.ToTensor()])
    return transform


def get_transform_cifar(mode: str, auto_augment: bool = False,
                        cutout: bool = False, cutout_length: int = None,
                        data_shape: list[int] = [3, 32, 32]) -> transforms.Compose:
    if mode != 'train':
        return transforms.Compose([transforms.ToTensor()])
    cutout_length = data_shape[-1] // 2 if cutout_length is None else cutout_length
    transform_list = [
        transforms.RandomCrop(data_shape[-2:], padding=data_shape[-1] // 8),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_augment:
        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    transform_list.append(transforms.ToTensor())
    if cutout:
        transform_list.append(Cutout(cutout_length))
    return transforms.Compose(transform_list)
