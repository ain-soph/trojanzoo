#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .environ import env
from .output import ansi
from .tensor import to_list

import torch
import torch.utils.data
import os
import tqdm
import tarfile
import zipfile
from typing import Union


def untar(file_path: str, target_path: str):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    tar = tarfile.open(file_path)
    names = tar.getnames()
    if env['tqdm']:
        names = tqdm(names)
    for name in names:
        tar.extract(name, path=target_path)
    if env['tqdm']:
        print('{upline}{clear_line}'.format(**ansi))
    tar.close()


def unzip(file_path: str, target_path: str):
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(target_path)


def uncompress(file_path: str, target_path: str, verbose: bool = True):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if verbose:
        print('{yellow}Uncompress file{reset}: '.format(**ansi), file_path)
    ext = os.path.splitext(file_path)[1]
    if ext in ['.zip']:
        unzip(file_path, target_path)
    elif ext in ['.tar', '.gz']:
        untar(file_path, target_path)
    else:
        raise NotImplementedError(f'{file_path=}')
    if verbose:
        print('{green}Uncompress finished{reset}: '.format(**ansi),
              f'{target_path}')
        print()


class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor = None, targets: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.targets = to_list(targets)
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index: Union[int, slice]) -> tuple[torch.Tensor, int]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def dataset_to_list(dataset: torch.utils.data.Dataset, label_only: bool = False) -> tuple[list, list[int]]:
    if label_only and 'targets' in dataset.__dict__.keys():
        return None, dataset.targets
    if 'data' in dataset.__dict__.keys() and 'targets' in dataset.__dict__.keys():
        return dataset.data, dataset.targets
    data, targets = zip(*dataset)
    if label_only:
        data = None
    else:
        data = list(data)
    targets = list(targets)
    return data, targets


def sample_batch(dataset: torch.utils.data.Dataset, batch_size: int = None,
                 idx: list[int] = None) -> tuple[list, list[int]]:
    if idx is None:
        assert len(dataset) >= batch_size
        idx = torch.randperm(len(dataset))[:batch_size]
    else:
        assert len(dataset) > max(idx)
    subset = torch.utils.data.Subset(dataset, idx)
    return dataset_to_list(subset)
