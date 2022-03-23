#!/usr/bin/env python3

# https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/datasets/DownsampledImageNet.py

##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os
import pickle
from PIL import Image
import numpy as np

from torchvision.datasets.utils import check_integrity
from torchvision.datasets import VisionDataset

from collections.abc import Callable


class DownsampledImageNet(VisionDataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    data_shape: list[int] = []
    train_list: list[list[str]] = []
    test_list: list[list[str]] = []

    def __init__(self, root: str, train: bool = True,
                 num_classes: int = None,
                 transform: None | Callable = None,
                 target_transform: None | Callable = None,
                 download: bool = False,
                 ) -> None:
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        self.num_classes = num_classes
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets: list[int] = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend([label - 1 for label in entry['labels']])
        self.data = np.vstack(self.data).reshape(-1, *self.data_shape)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if num_classes is not None:
            assert 0 < num_classes < 1000, f'{num_classes=}'
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if L < num_classes:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets

    def __repr__(self) -> str:
        lines = super().__repr__().split('\n')
        lines.insert(1, f'Number of Classes: {self.num_classes}')
        return '\n'.join(lines)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for filename, md5 in (self.train_list + self.test_list):
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class ImageNet16(DownsampledImageNet):
    data_shape: list[int] = [3, 16, 16]
    train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'],
    ]
    test_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]


class ImageNet32(DownsampledImageNet):
    data_shape: list[int] = [3, 32, 32]
    train_list = [
        ['train_data_batch_1', 'dd6683a336ab645d336f7b47c67d8456'],
        ['train_data_batch_2', 'b9b1f5ad237638a41e80944fc03b42af'],
        ['train_data_batch_3', 'e33306a0f4234b5c5812a32845bfbd04'],
        ['train_data_batch_4', '4db6d81c460a5170a03ef03e1ddd21dd'],
        ['train_data_batch_5', 'baa0f1d06099f55e5fe50d43fca4d334'],
        ['train_data_batch_6', 'bba3bc6e83e0703df1ec7e277e0b6800'],
        ['train_data_batch_7', '48dd0fd4511b39d518dbabba2f5b36b7'],
        ['train_data_batch_8', '46583e44f926baaeacbf8c333dc6110f'],
        ['train_data_batch_9', '7c82c36ce028b457204f1080d1793669'],
        ['train_data_batch_10', '14e3a6ce19d584ba97208a1d9f551e63'],
    ]
    test_list = [
        ['val_data', '4836a1eec28cd4476eb017126cd0f059'],
    ]
