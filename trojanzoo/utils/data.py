# -*- coding: utf-8 -*-

from .environ import env
from .output import ansi
from .tensor import to_tensor

import torch
from trojanzoo.datasets import Dataset
from torchvision.datasets import VisionDataset
import numpy as np
import PIL.Image as Image
import tarfile
import zipfile
import struct
import io
import os
import tqdm
from typing import Any, List, Tuple, Union


def untar(file_path, target_path):
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


def unzip(file_path, target_path):
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(target_path)


def uncompress(file_path: List[str], target_path: str, verbose=True):
    if isinstance(file_path, str):
        file_path = [file_path]
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for _file in file_path:
        if verbose:
            print('Uncompress file: ', _file)
        ext = os.path.splitext(_file)[1]
        if ext in['.zip']:
            unzip(_file, target_path)
        elif ext in ['.tar', '.gz']:
            untar(_file, target_path)
        else:
            raise TypeError('Not Compression File path: %s' % _file)
        if verbose:
            print('Uncompress finished at: ', target_path)
            print()


def convert_dataset_to_tensor(dataset: VisionDataset, **kwargs):
    if 'data' in dataset.__dict__.keys() and 'targets' in dataset.__dict__.keys():
        return to_tensor(dataset.data, **kwargs), to_tensor(dataset.targets, **kwargs)
    raise NotImplementedError('TODO')


class TensorListDataset(Dataset):
    def __init__(self, data: torch.FloatTensor = None, targets: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.targets = targets

    def __getitem__(self, index: Union[int, slice]) -> Tuple[torch.FloatTensor, int]:
        return self.data[index], int(self.targets[index])

    def __len__(self):
        return len(self.data)


class MemoryDataset(VisionDataset):
    def __init__(self, data: np.ndarray = None, targets: List[int] = None,
                 root: str = None, **kwargs):
        super().__init__(root=root, **kwargs)
        self.data = data
        self.targets = targets
        if data is None and os.path.isfile(root) and root.endswith('.npz'):
            _dict = np.load(root)
            self.data = _dict['data']
            self.targets = list(_dict['targets'])

    def __getitem__(self, index: Union[int, slice]) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class ZipFolder(VisionDataset):
    pass


# https://github.com/koenvandesande/vision/blob/read_zipped_data/torchvision/datasets/utils.py
class ZipLookup(object):
    def __init__(self, filename):
        self.root_zip_filename = filename
        self.root_zip_lookup = {}

        with zipfile.ZipFile(filename, "r") as root_zip:
            for info in root_zip.infolist():
                if info.filename[-1] == '/':
                    # skip directories
                    continue
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError("Only uncompressed ZIP file supported: " + info.filename)
                if info.compress_size != info.file_size:
                    raise ValueError("Must be the same when uncompressed")
                self.root_zip_lookup[info.filename] = (info.header_offset, info.compress_size)

    def __getitem__(self, path):
        z = open(self.root_zip_filename, "rb")
        header_offset, size = self.root_zip_lookup[path]

        z.seek(header_offset)
        fheader = z.read(zipfile.sizeFileHeader)
        fheader = struct.unpack(zipfile.structFileHeader, fheader)
        offset = header_offset + zipfile.sizeFileHeader + fheader[zipfile._FH_FILENAME_LENGTH] + \
            fheader[zipfile._FH_EXTRA_FIELD_LENGTH]

        z.seek(offset)
        f = io.BytesIO(z.read(size))
        f.name = path
        z.close()
        return f

    def keys(self):
        return self.root_zip_lookup.keys()
