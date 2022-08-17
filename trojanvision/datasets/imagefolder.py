#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet
from trojanvision.utils.dataset import ZipFolder
from trojanvision.environ import env
from trojanzoo.utils.output import ansi, prints, output_iter

import torchvision.datasets as datasets
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive,
                                        extract_archive)
import numpy as np
import zipfile
import os
import glob
import shutil
from tqdm import tqdm

from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    pass


class ImageFolder(ImageSet):
    r"""Image folder class which inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        :any:`torchvision.datasets.ImageFolder`

    Attributes:
        url (dict[str, str]): links to data files.
        ext (dict[str, str]): Map from mode to downloaded file extension.
        md5 (dict[str, str]): Map from mode to downloaded file md5.
        org_folder_name (dict[str, str]):
            Map from mode to extracted folder name of downloaded file.

        data_format (str): File format of dataset.

            * ``'folder'`` (default)
            * ``'tar'``
            * ``'zip'``

        memory (bool): Whether to put all dataset into memory at initialization.
            Defaults to ``False``.
    """

    name: str = 'imagefolder'
    url: dict[str, str] = {}
    ext = {'train': '.zip', 'valid': '.zip', 'test': '.zip'}    # TODO: Use Param?  # noqa
    md5: dict[str, str] = {}
    org_folder_name: dict[str, str] = {}

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        r"""Add image dataset arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete dataset class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.

        See Also:
            :meth:`trojanvision.datasets.ImageSet.add_argument()`
        """
        super().add_argument(group)
        group.add_argument('--data_format', choices=['folder', 'tar', 'zip'],
                           help='file format of dataset.'
                           '(zip is using ZIP_STORED)')
        group.add_argument('--memory', action='store_true',
                           help='put all dataset into memory at initialization.')
        return group

    def __init__(self, data_format: str = 'folder',
                 memory: bool = False, **kwargs):
        self.data_format: str = data_format
        self.memory: bool = memory
        super().__init__(**kwargs)
        self.param_list['imagefolder'] = ['data_format', 'memory', 'org_folder_name',
                                          'url']
        self.class_names = self.get_class_names()
        self.num_classes = self.num_classes or len(self.class_names)

    def initialize(self, *args, **kwargs):
        r"""You could use this method to transform across different :attr:`data_format`."""
        if self.data_format == 'folder' or \
                not self.check_files(data_format='folder'):
            self.initialize_folder(*args, **kwargs)
        if self.data_format == 'zip':
            self.initialize_zip(*args, **kwargs)

    def initialize_folder(self, **kwargs):
        print('initialize folder')
        mode_list = ['train']
        if self.valid_set and 'valid' in self.url.keys():
            mode_list.append('valid')
        for mode in mode_list:
            zip_path = os.path.join(self.folder_path,
                                    f'{self.name}_{mode}_store.zip')
            if os.path.isfile(zip_path):
                print('{yellow}Uncompress file{reset}: '.format(**ansi),
                      zip_path)
                extract_archive(from_path=zip_path, to_path=self.folder_path)
                print('{green}Uncompress finished{reset}: '.format(**ansi),
                      zip_path)
                print()
                continue
            self.download_and_extract_archive(mode=mode)
            os.rename(os.path.join(self.folder_path,
                                   self.org_folder_name[mode]),
                      os.path.join(self.folder_path, mode))
            try:
                dirname = os.path.dirname(self.org_folder_name[mode])
                if dirname:
                    shutil.rmtree(os.path.join(self.folder_path, dirname))
            except FileNotFoundError:
                pass

    def download_and_extract_archive(self, mode: str):
        file_name = f'{self.name}_{mode}{self.ext[mode]}'
        file_path = os.path.normpath(os.path.join(self.folder_path, file_name))
        md5 = self.md5.get(mode)
        if not check_integrity(file_path, md5=md5):
            prints('{yellow}Downloading Dataset{reset} '.format(**ansi),
                   f'{self.name} {mode:5s}: {file_path}', indent=10)
            download_and_extract_archive(url=self.url[mode],
                                         download_root=self.folder_path,
                                         extract_root=self.folder_path,
                                         filename=file_name, md5=md5)
            prints('{upline}{clear_line}'.format(**ansi), indent=10)
        else:
            prints('{yellow}File Already Exists{reset}: '.format(**ansi),
                   file_path, indent=10)
            extract_archive(from_path=file_path, to_path=self.folder_path)

    def initialize_zip(self, mode_list: list[str] = None, **kwargs):
        mode_list = mode_list or [mode for mode in ['train', 'valid', 'test']
                                  if os.path.isdir(
                                      os.path.join(self.folder_path, mode))]
        for mode in mode_list:
            src_path = os.path.normpath(os.path.join(self.folder_path, mode))
            dst_path = os.path.join(self.folder_path,
                                    f'{self.name}_{mode}_store.zip')
            assert os.path.isdir(src_path)
            if not os.path.exists(dst_path):
                print('{yellow}initialize zip{reset}: '.format(**ansi),
                      dst_path)
                ZipFolder.initialize_from_folder(root=src_path,
                                                 zip_path=dst_path)
                print('{green}initialize zip finish{reset}'.format(**ansi))

    def _get_org_dataset(self, mode: str, data_format: str = None,
                         **kwargs) -> datasets.DatasetFolder:
        data_format = data_format or self.data_format
        root = os.path.join(self.folder_path, mode)
        DatasetClass = datasets.ImageFolder
        if data_format == 'zip':
            root = os.path.join(self.folder_path,
                                f'{self.name}_{mode}_store.zip')
            DatasetClass = ZipFolder
            if 'memory' not in kwargs.keys():
                kwargs['memory'] = self.memory
        return DatasetClass(root=root, **kwargs)

    def get_class_names(self) -> list[str]:
        if hasattr(self, 'class_names'):
            return getattr(self, 'class_names')
        dataset: datasets.ImageFolder = self.get_org_dataset('train')
        return dataset.classes

    def sample(self, child_name: str = None,
               class_dict: dict[str, list[str]] = None,
               sample_num: int = None,
               method='zip'):
        r"""Sample a subset image folder dataset.

        Args:
            child_name (str): Name of child subset.
                Defaults to ``'{self.name}_sample{sample_num}'``
            class_dict (dict[str, list[str]] | None):
                Map from new class name to list of old class names.
                If ``None``, use :attr:`sample_num` to
                random sample a subset (1 to 1).
                Defaults to ``None``.
            sample_num (int | None):
                The number of subset classes to sample
                if :attr:`class_dict` is ``None``.
                Defaults to ``None``.
            method (str): :attr:`data_format` of new subset to save.
                Defaults to ``'.zip'``.
        """
        if sample_num is None:
            assert class_dict
            sample_num = len(class_dict)
        if child_name is None and sample_num is not None:
            child_name = f'{self.name}_sample{sample_num:d}'
        src_path = self.folder_path
        dst_path = os.path.normpath(os.path.join(
            os.path.dirname(self.folder_path), child_name))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        print('{yellow}src path{reset}: '.format(**ansi), src_path)
        print('{yellow}dst path{reset}: '.format(**ansi), dst_path)

        mode_list = [mode for mode in ['train', 'valid', 'test']
                     if os.path.isdir(os.path.join(src_path, mode))]
        if method == 'zip':
            zip_path_list: list[str] = glob.glob(
                os.path.join(src_path, '*_store.zip'))
            mode_list = [os.path.basename(zip_path).removeprefix(
                self.name).removesuffix('_store.zip')
                for zip_path in zip_path_list]

        src2dst_dict: dict[str, str] = {}
        if class_dict is None:
            assert sample_num
            idx_list: np.ndarray = np.arange(self.num_classes)
            np.random.seed(env['data_seed'])
            np.random.shuffle(idx_list)
            idx_list = idx_list[:sample_num]
            mode = mode_list[0]
            class_list: list[str] = []
            match method:
                case 'zip':
                    zip_path = os.path.join(src_path,
                                            f'{self.name}_{mode}_store.zip')
                    with zipfile.ZipFile(zip_path, 'r',
                                         compression=zipfile.ZIP_STORED
                                         ) as src_zip:
                        name_list = src_zip.namelist()
                    for name in name_list:
                        name_dir, name_base = os.path.split(os.path.dirname(name))
                        if name_dir == mode:
                            class_list.append(name_base)
                case 'folder':
                    folder_path = os.path.join(src_path, f'{mode}')
                    class_array: np.ndarray = np.array(
                        os.listdir(folder_path))[idx_list]
                    class_list = class_array.tolist()
                    class_list = [_dir for _dir in class_list
                                  if os.path.isdir(os.path.join(
                                      folder_path, _dir))]
            class_list.sort()
            class_array = np.array(class_list)[idx_list]
            class_list = class_array.tolist()
            for class_name in class_list:
                src2dst_dict[class_name] = class_name
        else:
            src2dst_dict = {src_class: dst_class
                            for src_class, dst_list in class_dict.items()
                            for dst_class in dst_list}
        src_class_list = src2dst_dict.keys()
        print(src2dst_dict)
        if method == 'zip':
            for mode in mode_list:
                print('{purple}mode: {0}{reset}'.format(mode, **ansi))
                assert mode in ['train', 'valid', 'test']
                dst_zip_path = os.path.join(dst_path,
                                            f'{child_name}_{mode}_store.zip')
                dst_zip = zipfile.ZipFile(dst_zip_path, 'w',
                                          compression=zipfile.ZIP_STORED)
                src_zip_path = os.path.join(src_path,
                                            f'{self.name}_{mode}_store.zip')
                src_zip = zipfile.ZipFile(src_zip_path, 'r',
                                          compression=zipfile.ZIP_STORED)
                _list = src_zip.namelist()
                if env['tqdm']:
                    _list = tqdm(_list, leave=False)
                for filename in _list:
                    if filename[-1] == '/':
                        continue
                    dirname, basename = os.path.split(filename)
                    mode_check, src_class = os.path.split(dirname)
                    if mode_check == mode and src_class in src_class_list:
                        print(filename)
                        dst_class = src2dst_dict[src_class]
                        dst_zip.writestr(f'{mode}/{dst_class}/{basename}',
                                         src_zip.read(filename))
                src_zip.close()
                dst_zip.close()
        elif method == 'folder':
            len_i = len(class_dict.keys())
            for mode in mode_list:
                print('{purple}{0}{reset}'.format(mode, **ansi))
                assert mode in ['train', 'valid', 'test']
                for i, dst_class in enumerate(class_dict.keys()):
                    if not os.path.exists(_path := os.path.join(dst_path,
                                                                mode,
                                                                dst_class)):
                        os.makedirs(_path)
                    prints('{blue_light}{0}{reset}'.format(dst_class, **ansi),
                           indent=10)
                    class_list = class_dict[dst_class]
                    len_j = len(class_list)
                    for j, src_class in enumerate(class_list):
                        _list = os.listdir(os.path.join(src_path,
                                                        mode,
                                                        src_class))
                        prints(output_iter(i + 1, len_i),
                               output_iter(j + 1, len_j),
                               f'dst: {dst_class:15s}    '
                               f'src: {src_class:15s}    '
                               f'image_num: {len(_list):>8d}',
                               indent=10)
                        if env['tqdm']:
                            _list = tqdm(_list, leave=False)
                        for _file in _list:
                            src_file_path = os.path.join(src_path, mode,
                                                         src_class, _file)
                            dst_file_path = os.path.join(dst_path, mode,
                                                         dst_class, _file)
                            shutil.copyfile(src_file_path, dst_file_path)
                        if env['tqdm']:
                            print('{upline}{clear_line}'.format(**ansi))
