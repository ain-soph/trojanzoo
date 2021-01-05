#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imageset import ImageSet
from trojanvision.utils.data import MemoryDataset, ZipFolder
from trojanvision.environ import env
from trojanzoo.utils.output import ansi, prints, output_iter
from trojanzoo.utils.data import uncompress, dataset_to_list

from torch.hub import download_url_to_file
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
                           help='folder, zip or numpy. (zip is using ZIP_STORED)')

    def __init__(self, data_format: str = 'folder', **kwargs):
        self.data_format: str = data_format
        if data_format == 'zip':
            raise NotImplementedError('data_format=zip Not supported! See reason at issues/42 on github.')
        super().__init__(**kwargs)
        self.param_list['imagefolder'] = ['data_format', 'url', 'org_folder_name']
        self.class_to_idx = self.get_class_to_idx()
        if self.class_to_idx is None and data_format == 'folder':
            self.class_to_idx = self.get_class_to_idx(check_folder=True)
        if self.num_classes is None:
            self.num_classes = len(self.class_to_idx)

    def middle_process(self):
        mode_list = ['valid']
        if self.data_format not in ['folder', 'zip']:
            mode_list = ['train', 'valid']
        self.data, self.targets = self.load_npz(mode_list=mode_list)

    def initialize(self, *args, **kwargs):
        if not self.check_files(data_format='folder'):
            self.initialize_folder(*args, **kwargs)
        if self.data_format == 'folder':
            self.initialize_npz(mode_list=['valid'])
        elif self.data_format == 'zip':
            self.initialize_zip(*args, **kwargs)
            self.initialize_npz(mode_list=['valid'])
        else:
            self.initialize_npz(*args, **kwargs)

    def initialize_folder(self, verbose: bool = True, img_type: str = '.jpg', **kwargs):
        print('initialize folder')
        mode_list: list[str] = ['train', 'valid'] if self.valid_set else ['train']
        self.class_to_idx = self.get_class_to_idx()
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for mode in mode_list:
            zip_path = os.path.join(self.folder_path, f'{self.name}_{mode}_store.zip')
            npz_path = os.path.join(self.folder_path, f'{self.name}_{mode}.npz')
            if os.path.isfile(zip_path):
                uncompress(file_path=zip_path, target_path=self.folder_path, verbose=verbose)
                continue
            elif os.path.isfile(npz_path):
                self.data, self.targets = self.load_npz()
                class_counters = [0] * self.num_classes
                # TODO: Parallel
                for image, target_class in self.data, self.targets:
                    image = Image.fromarray(image)
                    class_name = idx_to_class[target_class]
                    _dir = os.path.join(self.folder_path, mode, class_name)
                    if not os.path.exists(_dir):
                        os.makedirs(_dir)
                    image.save(os.path.join(_dir, f'{class_counters[target_class]}{img_type}'))
                    class_counters[target_class] += 1
                continue
            file_path = self.download(mode=mode, url=self.url[mode])
            uncompress(file_path=file_path, target_path=self.folder_path, verbose=verbose)
            os.rename(os.path.join(self.folder_path, self.org_folder_name[mode]),
                      os.path.join(self.folder_path, mode))
            try:
                shutil.rmtree(os.path.join(self.folder_path, os.path.dirname(self.org_folder_name[mode])))
            except FileNotFoundError:
                pass

    def initialize_zip(self, mode_list: list[str] = ['train', 'valid'], **kwargs):
        if not self.valid_set:
            mode_list.remove('valid')
        for mode in mode_list:
            dst_path = os.path.join(self.folder_path, f'{self.name}_{mode}_store.zip')
            if not os.path.exists(dst_path):
                print('{yellow}initialize zip{reset}: '.format(**ansi), dst_path)
                src_path = os.path.normpath(os.path.join(self.folder_path, mode))
                with zipfile.ZipFile(dst_path, mode='w', compression=zipfile.ZIP_STORED) as zf:
                    for root, dirs, files in os.walk(src_path):
                        _dir = root.removeprefix(os.path.join(self.folder_path, ''))
                        for _file in files:
                            org_path = os.path.join(root, _file)
                            zip_path = os.path.join(_dir, _file)
                            zf.write(org_path, zip_path)
                print('{green}initialize zip finish{reset}'.format(**ansi))

    def initialize_npz(self, mode_list: list[str] = ['train', 'valid'],
                       transform: transforms.Lambda = transforms.Lambda(lambda x: np.array(x)),
                       **kwargs):
        if not self.valid_set:
            mode_list.remove('valid')
        json_path = os.path.join(self.folder_path, 'class_to_idx.json')
        for mode in mode_list:
            npz_path = os.path.join(self.folder_path, f'{self.name}_{mode}.npz')
            if not os.path.exists(npz_path):
                print('{yellow}initialize npz{reset}: '.format(**ansi), npz_path)
                dataset: datasets.ImageFolder = self.get_org_dataset(mode, transform=transform, data_format='folder')
                data, targets = dataset_to_list(dataset)
                data = np.stack(data)
                np.savez(npz_path, data=data, targets=targets)
                with open(json_path, 'w') as f:
                    json.dump(dataset.class_to_idx, f)
                print('{green}initialize npz finish{reset}: '.format(**ansi))

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default',
                        data_format: str = None, **kwargs) -> Union[datasets.ImageFolder, MemoryDataset]:
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        if data_format is None:
            data_format = self.data_format
            # if mode == 'valid':
            #     data_format = 'numpy'
        root = os.path.join(self.folder_path, f'{self.name}_{mode}.npz')
        if data_format == 'folder':
            root = os.path.join(self.folder_path, mode)
        elif data_format == 'zip':
            root = os.path.join(self.folder_path, f'{self.name}_{mode}_store.zip')
        DatasetClass = datasets.VisionDataset
        if data_format == 'folder':
            DatasetClass = datasets.ImageFolder
        elif data_format == 'zip':
            DatasetClass = ZipFolder
        elif not ('data' in self.__dict__.keys()):
            raise RuntimeError()   # TODO
        else:
            DatasetClass = MemoryDataset
            kwargs['data'] = self.data[mode]
            kwargs['targets'] = self.targets[mode]
        return DatasetClass(root=root, transform=transform, **kwargs)

    def load_npz(self, mode_list: list[str] = ['train', 'valid']) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
        if not self.valid_set:
            mode_list.remove('valid')
        data = {}
        targets = {}
        for mode in mode_list:
            npz_path = os.path.join(self.folder_path, f'{self.name}_{mode}.npz')
            _dict = np.load(npz_path)
            data[mode] = _dict['data']
            targets[mode] = list(_dict['targets'])
        return data, targets

    def get_class_to_idx(self, file_path: str = None, check_folder=False) -> dict[str, int]:
        if file_path is None:
            file_path = os.path.join(self.folder_path, 'class_to_idx.json')
        if os.path.exists(file_path):
            with open(file_path) as fp:
                result: dict[str, int] = json.load(fp)
            return result
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
                file_path = os.path.normpath(os.path.join(folder_path, file_name))
        if not os.path.exists(file_path):
            print('{yellow}Downloading Dataset{reset} '.format(**ansi),
                  f'{self.name} {mode:5s}: {file_path}')
            download_url_to_file(url, file_path)
            print('{upline}{clear_line}'.format(**ansi))
        else:
            prints('{yellow}File Already Exists{reset}: '.format(**ansi), file_path, indent=10)
        return file_path

    def sample(self, child_name: str = None, class_dict: dict = None, sample_num: int = None, verbose=True):
        if sample_num is None:
            assert class_dict
            sample_num = len(class_dict)
        if child_name is None:
            child_name = self.name + '_sample%d' % sample_num
        src_path = self.folder_path
        mode_list = [_dir for _dir in os.listdir(
            src_path) if os.path.isdir(src_path + _dir) and _dir[0] != '.']
        dst_path = os.path.normpath(os.path.join(os.path.dirname(self.folder_path), child_name))
        if verbose:
            print('{yellow}src path{reset}: '.format(**ansi), src_path)
            print('{yellow}dst path{reset}: '.format(**ansi), dst_path)
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
                print('{purple}{0}{reset}'.format(src_mode, **ansi))
            assert src_mode in ['train', 'valid', 'test', 'val']
            dst_mode = 'valid' if src_mode == 'val' else src_mode
            for i, dst_class in enumerate(class_dict.keys()):
                if not os.path.exists(os.path.join(dst_path, dst_mode, dst_class)):
                    os.makedirs(os.path.join(dst_path, dst_mode, dst_class))
                prints('{blue_light}{0}{reset}'.format(dst_class, **ansi), indent=10)
                class_list = class_dict[dst_class]
                len_j = len(class_list)
                for j, src_class in enumerate(class_list):
                    _list = os.listdir(os.path.join(src_path, src_mode, src_class))
                    prints(output_iter(i + 1, len_i) + output_iter(j + 1, len_j) +
                           f'dst: {dst_class:15s}    src: {src_class:15s}    image_num: {len(_list):>8d}', indent=10)
                    if env['tqdm']:
                        _list = tqdm(_list)
                    for _file in _list:
                        shutil.copyfile(os.path.join(src_path, src_mode, src_class, _file),
                                        os.path.join(dst_path, dst_mode, dst_class, _file))
                    if env['tqdm']:
                        print('{upline}{clear_line}'.format(**ansi))
