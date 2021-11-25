#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms
import random
from typing import Union  # TODO: python 3.10


__all__ = ['Cutout', 'get_transform_bit',
           'get_transform_imagenet', 'get_transform_cifar']


class Cutout:
    def __init__(self, length: int,
                 fill_values: Union[float, torch.Tensor] = 0.0):
        self.length = length
        self.fill_values = fill_values

    def __call__(self, img: torch.Tensor):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=torch.bool, device=img.device)
        y = random.randint(0, h)
        x = random.randint(0, w)
        y1 = max(y - self.length // 2, 0)
        y2 = min(y + self.length // 2, h)
        x1 = max(x - self.length // 2, 0)
        x2 = min(x + self.length // 2, w)
        mask[y1: y2, x1: x2] = False
        return (mask * img + ~mask * self.fill_values).detach()


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


def get_transform_imagenet(mode: str, use_tuple: bool = False,
                           auto_augment: bool = False) -> transforms.Compose:
    if mode == 'train':
        transform_list = [
            transforms.RandomResizedCrop((224, 224) if use_tuple else 224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # noqa
        ]
        if auto_augment:
            transform_list.append(transforms.AutoAugment(
                transforms.AutoAugmentPolicy.IMAGENET))
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
                        data_shape: list[int] = [3, 32, 32]
                        ) -> transforms.Compose:
    if mode != 'train':
        return transforms.Compose([transforms.ToTensor()])
    cutout_length = cutout_length or data_shape[-1] // 2
    transform_list = [
        transforms.RandomCrop(data_shape[-2:], padding=data_shape[-1] // 8),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_augment:
        transform_list.append(transforms.AutoAugment(
            transforms.AutoAugmentPolicy.CIFAR10))
    transform_list.append(transforms.ToTensor())
    if cutout:
        transform_list.append(Cutout(cutout_length))
    return transforms.Compose(transform_list)
