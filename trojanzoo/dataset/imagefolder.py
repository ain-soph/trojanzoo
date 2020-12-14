# -*- coding: utf-8 -*-

from .imageset import ImageSet
from trojanzoo.environ import env
from trojanzoo.utils.output import ansi, prints, output_iter
from trojanzoo.utils.data import uncompress

from torch.hub import download_url_to_file
import torchvision.datasets as datasets
import numpy as np
import os
import shutil
from tqdm import tqdm
from typing import List, Dict, Union


class ImageFolder(ImageSet):

    name: str = 'imagefolder'
    url: Dict[str, str] = {}
    org_folder_name: Dict[str, str] = {}

    def __init__(self, data_format: str = 'folder', in_memory: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.param_list['imagefolder'] = ['url', 'org_folder_name']
        self.class_to_idx: Dict[str, int] = \
            self.get_org_dataset('train').class_to_idx
        if self.num_classes is None:
            self.num_classes = len(self.class_to_idx)
        self.data_format: str = data_format
        self.in_memory: bool = in_memory

    def initialize(self, verbose=True, **kwargs):
        if self.data_format == 'folder':
            self.initialize_folder(verbose=verbose, **kwargs)
        elif self.data_format in['numpy', 'np', 'torch', 'pytorch', 'tensor']:
            pass

    def initialize_folder(self, verbose=True, **kwargs):
        file_path = self.download()
        uncompress(file_path=file_path.values(),
                   target_path=self.folder_path + self.name, verbose=verbose)
        mode_list: List[str] = ['train']
        if self.valid_set:
            mode_list.append('valid')
        for mode in mode_list:
            os.rename(self.folder_path + self.name + f'/{self.org_folder_name[mode]}/',
                      self.folder_path + self.name + f'/{mode}/')
            if '/' in self.org_folder_name[mode]:
                shutil.rmtree(self.folder_path + self.name + '/' +
                              self.org_folder_name[mode].split('/')[0])

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default', **kwargs) -> datasets.ImageFolder:
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        return datasets.ImageFolder(root=self.folder_path + self.name + f'/{mode}/',
                                    transform=transform, **kwargs)

    def download(self, url: Dict[str, str] = None, file_path: str = None,
                 folder_path: str = None, file_name: str = None, file_ext: str = 'zip'):
        if url is None:
            url = self.url
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if file_name is None:
                file_name = {}
                file_path = {}
                file_name['train'] = self.name + '_train.' + file_ext
                file_path['train'] = folder_path + file_name['train']
                if self.valid_set:
                    file_name['valid'] = self.name + '_valid.' + file_ext
                    file_path['valid'] = folder_path + file_name['valid']
        print('Downloading Dataset %s' % self.name)
        for mode in file_path.keys():
            prints(mode, ' ' * 10, file_path[mode], indent=10)
            if not os.path.exists(file_path[mode]):
                download_url_to_file(url[mode], file_path[mode])
                print('{upline}{clear_line}'.format(**ansi), end='')
            else:
                prints('File Already Exists: ', file_path[mode], indent=20)
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
