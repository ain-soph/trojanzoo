# -*- coding: utf-8 -*-

from .environ import env
from .output import ansi
from .tensor import to_list

import torch
from torch.utils.data import Dataset
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
        print('{upline}{clear_line}'.format(**ansi), end='')
    tar.close()


def unzip(file_path: str, target_path: str):
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(target_path)


def uncompress(file_path: str, target_path: str, verbose: bool = True):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if verbose:
        print('Uncompress file: ', file_path)
    ext = os.path.splitext(file_path)[1]
    if ext in ['.zip']:
        unzip(file_path, target_path)
    elif ext in ['.tar', '.gz']:
        untar(file_path, target_path)
    else:
        raise NotImplementedError(f'{file_path=}')
    if verbose:
        print(f'Uncompress finished: {target_path}')
        print()


class TensorListDataset(Dataset):
    def __init__(self, data: torch.Tensor = None, targets: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.targets = to_list(targets)
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index: Union[int, slice]) -> tuple[torch.Tensor, int]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)
