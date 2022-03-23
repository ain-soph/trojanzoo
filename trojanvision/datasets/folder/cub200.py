#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
from trojanzoo.utils.output import ansi, prints

from torchvision.datasets.utils import download_file_from_google_drive, extract_archive, check_integrity
import os
import shutil
import pandas as pd

from trojanvision import __file__ as root_file
root_dir = os.path.dirname(root_file)


class CUB200(ImageFolder):
    r"""CUB200 dataset introduced by Peter Welinder in 2010.
    It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * paper: `Caltech-UCSD Birds 200`_
        * website: http://www.vision.caltech.edu/visipedia/CUB-200.html

    Attributes:
        name (str): ``'cub200'``
        num_classes (int): ``200``
        data_shape (list[int]): ``[3, 224, 224]``
        valid_set (bool): ``False``

    .. _Caltech-UCSD Birds 200:
        http://www.vision.caltech.edu/visipedia/papers/WelinderEtal10_CUB-200.pdf
    """
    name = 'cub200'
    num_classes = 200
    valid_set = False
    # http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
    url = {'train': '1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx'}
    ext = {'train': '.tgz'}
    md5 = {'train': '2bbe304ef1aa3ddb6094aa8f53487cf2'}

    org_folder_name = {'train': 'images'}

    def download_and_extract_archive(self, mode: str):
        file_name = f'{self.name}_{mode}{self.ext[mode]}'
        file_path = os.path.normpath(os.path.join(self.folder_path, file_name))
        md5 = self.md5.get(mode)
        if not check_integrity(file_path, md5=md5):
            prints('{yellow}Downloading Dataset{reset} '.format(**ansi),
                   f'{self.name} {mode:5s}: {file_path}', indent=10)
            download_file_from_google_drive(file_id=self.url[mode],
                                            root=self.folder_path, filename=file_name, md5=md5)
            print('{upline}{clear_line}'.format(**ansi))
        else:
            prints('{yellow}File Already Exists{reset}: '.format(**ansi), file_path, indent=10)
        extract_archive(from_path=file_path, to_path=self.folder_path)

    def initialize_folder(self, **kwargs):
        super().initialize_folder(**kwargs)
        self.split()

    def split(self):
        # Remove useless files
        os.remove(os.path.join(self.folder_path, '._images'))
        dirpath = os.path.join(self.folder_path, 'train')
        for fpath in os.listdir(dirpath):
            path = os.path.join(dirpath, fpath)
            if os.path.isfile(path):
                os.remove(path)

        # Split Train and Valid Set
        txt_path = os.path.normpath(os.path.join(root_dir, 'data', 'cub200', 'test.txt'))
        file_list: list[str] = []
        with open(txt_path, 'r') as fp:
            file_list = fp.read().split('\n')[:-1]
        src_dir = os.path.join(self.folder_path, 'train')
        dst_dir = os.path.join(self.folder_path, 'valid')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for fpath in file_list:
            src_path = os.path.join(src_dir, fpath)
            dst_path = os.path.join(dst_dir, fpath)
            dir_name = os.path.dirname(dst_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            shutil.move(src_path, dst_path)


class CUB200_2011(CUB200):
    r"""CUB200_2011 dataset. It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * paper: `The Caltech-UCSD Birds-200-2011 Dataset`_
        * website: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    Attributes:
        name (str): ``'cub200_2011'``
        num_classes (int): ``200``
        data_shape (list[int]): ``[3, 224, 224]``
        valid_set (bool): ``False``

    .. _The Caltech-UCSD Birds-200-2011 Dataset:
        http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf
    """

    name = 'cub200_2011'
    # http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    url = {'train': '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'}
    ext = {'train': '.tgz'}
    md5 = {'train': '97eceeb196236b17998738112f37df78'}

    org_folder_name = {'train': 'CUB_200_2011/images'}

    def split(self):
        # Split Train and Valid Set
        src_dir = os.path.join(self.folder_path, 'total')
        dst_dir = {'train': os.path.join(self.folder_path, 'train'),
                   'valid': os.path.join(self.folder_path, 'valid')}
        os.rename(dst_dir['train'], src_dir)
        os.remove(os.path.join(self.folder_path, 'attributes.txt'))

        images = pd.read_csv(os.path.join(root_dir, 'data', 'cub200_2011', 'images.txt'),
                             sep=' ', names=['img_id', 'filepath'])
        train_test_split = pd.read_csv(os.path.join(root_dir, 'data', 'cub200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(train_test_split, on='img_id')
        file_dict: dict[str, list[str]] = {
            'train': data[data.is_training_img == 1]['filepath'].tolist(),
            'valid': data[data.is_training_img == 0]['filepath'].tolist(),
        }
        for mode in ['train', 'valid']:
            for fpath in file_dict[mode]:
                src_path = os.path.join(src_dir, fpath)
                dst_path = os.path.join(dst_dir[mode], fpath)
                dir_name = os.path.dirname(dst_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                shutil.move(src_path, dst_path)
        shutil.rmtree(src_dir)
