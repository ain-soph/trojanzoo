# -*- coding: utf-8 -*-

from .imageset import ImageSet
from trojanvision.utils.data import MemoryDataset, ZipFolder
from trojanvision.environ import env
from trojanzoo.utils.output import ansi, prints, output_iter
from trojanzoo.utils.data import uncompress

from torch.hub import download_url_to_file
import torchvision.datasets as datasets
import numpy as np
import PIL.Image as Image
import json
import os
import shutil
import zipfile
import argparse
from tqdm import tqdm
from typing import Union


class ImageFolder(ImageSet):

    name: str = 'imagefolder'
    url: dict[str, str] = {}
    org_folder_name: dict[str, str] = {}

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--data_format', dest='data_format', type=str,
                           help='folder, zip or numpy. (zip is using ZIP_STOREED)')

    def __init__(self, data_format: str = 'folder', **kwargs):
        self.data_format: str = data_format
        super().__init__(**kwargs)
        self.param_list['imagefolder'] = ['data_format', 'url', 'org_folder_name']
        self.class_to_idx = self.get_class_to_idx()
        if self.class_to_idx is None and data_format == 'folder':
            self.class_to_idx = self.get_class_to_idx(check_folder=True)
        if self.num_classes is None:
            self.num_classes = len(self.class_to_idx)
        if data_format not in ['folder', 'zip']:
            self.data, self.targets = self.load_npz()

    def initialize(self, *args, **kwargs):
        if self.data_format == 'folder' or not self.check_files(data_format='folder'):
            self.initialize_folder(*args, **kwargs)
        if self.data_format == 'folder':
            return
        elif self.data_format == 'zip':
            self.initialize_zip(*args, **kwargs)
        else:
            self.initialize_npz(*args, **kwargs)

    def initialize_folder(self, verbose: bool = True, img_type: str = '.jpg', **kwargs):
        mode_list: list[str] = ['train', 'valid'] if self.valid_set else ['train']
        self.class_to_idx = self.get_class_to_idx()
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for mode in mode_list:
            zip_path = self.folder_path + self.name + f'/{self.name}_{mode}_store.zip'
            npz_path = self.folder_path + self.name + f'/{self.name}_{mode}.npz'
            if os.path.isfile(zip_path):
                uncompress(file_path=zip_path, target_path=self.folder_path + self.name, verbose=verbose)
                continue
            elif os.path.isfile(npz_path):
                self.data, self.targets = self.load_npz()
                class_counters = [0] * self.num_classes
                # TODO: Parallel
                for image, target_class in self.data, self.targets:
                    image = Image.fromarray(image)
                    class_name = idx_to_class[target_class]
                    _dir = self.folder_path + self.name + f'/{mode}/{class_name}/'
                    if not os.path.exists(_dir):
                        os.makedirs(_dir)
                    image.save(_dir + f'{class_counters[target_class]}{img_type}')
                    class_counters[target_class] += 1
                continue
            file_path = self.download(mode=mode)
            uncompress(file_path=file_path, target_path=self.folder_path + self.name, verbose=verbose)
            os.rename(self.folder_path + self.name + f'/{self.org_folder_name[mode]}/',
                      self.folder_path + self.name + f'/{mode}/')
            if '/' in self.org_folder_name[mode]:
                shutil.rmtree(self.folder_path + self.name + '/' +
                              self.org_folder_name[mode].split('/')[0])

    def initialize_zip(self, **kwargs):
        mode_list: list[str] = ['train', 'valid'] if self.valid_set else ['train']
        for mode in mode_list:
            src_path = self.folder_path + self.name + f'/{mode}/'
            dst_path = self.folder_path + self.name + f'/{self.name}_{mode}_store.zip'
            with open(zipfile.ZipFile(dst_path, mode='w', compression=zipfile.ZIP_STOREED)) as zf:
                for root, dirs, files in os.walk(src_path):
                    _dir = root.removeprefix(self.folder_path + self.name + '/')
                    for _file in files:
                        org_path = os.path.join(root, _file)
                        zip_path = os.path.join(_dir, _file)
                        zf.write(org_path, zip_path)

    def initialize_npz(self, **kwargs):
        mode_list: list[str] = ['train', 'valid'] if self.valid_set else ['train']
        json_path = self.folder_path + self.name + '/class_to_idx.json'
        for mode in mode_list:
            dataset: ImageFolder = self.get_org_dataset(mode, transform=None, data_format='folder')
            data, targets = self.to_memory(dataset)
            npz_path = self.folder_path + self.name + f'/{mode}.npz'
            np.savez(npz_path, data=data, targets=targets)
            with open(json_path, 'w') as f:
                json.dump(dataset.class_to_idx, f)

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default',
                        data_format: str = None, **kwargs) -> Union[datasets.ImageFolder, MemoryDataset]:
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        if data_format is None:
            data_format = self.data_format
        root = self.folder_path + self.name + f'/{mode}.npz'
        if data_format == 'folder':
            root = self.folder_path + self.name + f'/{mode}/'
        elif data_format == 'zip':
            root = self.folder_path + self.name + f'/{self.name}_{mode}_store.zip'
        DatasetClass = datasets.VisionDataset
        if data_format == 'folder':
            DatasetClass = datasets.ImageFolder
        elif data_format == 'zip':
            DatasetClass = ZipFolder
        elif not ('data' in self.__dict__.keys()):
            raise Exception()   # TODO
        else:
            DatasetClass = MemoryDataset
            kwargs['data'] = self.data[mode]
            kwargs['targets'] = self.targets[mode]
        return DatasetClass(root=root, transform=transform, **kwargs)

    def load_npz(self) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
        data = {}
        targets = {}
        mode_list: list[str] = ['train', 'valid'] if self.valid_set else ['train']
        for mode in mode_list:
            npz_path = self.folder_path + self.name + f'/{mode}.npz'
            _dict = np.load(npz_path)
            data[mode] = _dict['data']
            targets[mode] = list(_dict['targets'])
        return data, targets

    def get_class_to_idx(self, file_path: str = None, check_folder=False) -> dict[str, int]:
        if file_path is None:
            file_path = self.folder_path + self.name + '/class_to_idx.json'
        if os.path.exists(file_path):
            return json.load(file_path)
        if check_folder:
            return self.get_org_dataset('train', data_format='folder').class_to_idx
        return super().get_class_to_idx()

    def download(self, mode: str, url: str, file_path: str = None,
                 folder_path: str = None, file_name: str = None, file_ext: str = 'zip') -> str:
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if file_name is None:
                file_name = f'{self.name}_{mode}.{file_ext}'
                file_path = folder_path + file_name
        if not os.path.exists(file_path[mode]):
            print(f'Downloading Dataset {self.name} {mode:5s}: {file_path}')
            download_url_to_file(url[mode], file_path[mode])
            print('{upline}{clear_line}'.format(**ansi), end='')
        else:
            prints('File Already Exists: ', file_path, indent=10)
        return file_path

    def sample(self, child_name: str = None, class_dict: dict = None, sample_num: int = None, verbose=True):
        if sample_num is None:
            assert class_dict
            sample_num = len(class_dict)
        if child_name is None:
            child_name = self.name + '_sample%d' % sample_num
        src_path = self.folder_path + self.name + '/'
        mode_list = [_dir for _dir in os.listdir(
            src_path) if os.path.isdir(src_path + _dir) and _dir[0] != '.']
        dst_path = env['data_dir'] + self.data_type + \
            '/{0}/data/{0}/'.format(child_name)
        if verbose:
            print('src path: ', src_path)
            print('dst path: ', dst_path)
        if class_dict is None:
            assert sample_num
            idx_list = np.arange(self.num_classes)
            np.random.seed(env['seed'])
            np.random.shuffle(idx_list)
            idx_list = idx_list[:sample_num]
            class_list = np.array(os.listdir(src_path + mode_list[0]))[idx_list]
            class_dict = {}
            for class_name in class_list:
                class_dict[class_name] = [class_name]
        if verbose:
            print(class_dict)

        len_i = len(class_dict.keys())
        for src_mode in mode_list:
            if verbose:
                print(src_mode)
            assert src_mode in ['train', 'valid', 'test', 'val']
            dst_mode = 'valid' if src_mode == 'val' else src_mode
            for i, dst_class in enumerate(class_dict.keys()):
                if not os.path.exists(dst_path + dst_mode + '/' + dst_class):
                    os.makedirs(dst_path + dst_mode + '/' + dst_class)
                prints(dst_class, indent=10)
                class_list = class_dict[dst_class]
                len_j = len(class_list)
                for j, src_class in enumerate(class_list):
                    _list = os.listdir(src_path + src_mode + '/' + src_class)
                    prints(output_iter(i + 1, len_i) + output_iter(j + 1, len_j) +
                           f'dst: {dst_class:15s}    src: {src_class:15s}    image_num: {len(_list):>8d}', indent=10)
                    if env['tqdm']:
                        _list = tqdm(_list)
                    for _file in _list:
                        shutil.copyfile(src_path + src_mode + '/' + src_class + '/' + _file,
                                        dst_path + dst_mode + '/' + dst_class + '/' + _file)
                    if env['tqdm']:
                        print('{upline}{clear_line}'.format(**ansi), end='')
