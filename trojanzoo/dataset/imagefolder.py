# -*- coding: utf-8 -*-

from .imageset import ImageSet
from trojanzoo.utils.os import uncompress
from trojanzoo.utils.output import ansi, prints, output_iter

import os
import shutil
import numpy as np
from tqdm import tqdm
import urllib.request
from typing import Union, Dict
import torchvision.datasets as datasets

from trojanzoo.config import Config
env = Config.env


class ImageFolder(ImageSet):

    name: str = 'imagefolder'
    url: Dict[str, str] = {}
    org_folder_name: Dict[str, str] = {}

    def __init__(self, num_classes=None, **kwargs):
        super().__init__(num_classes=None, **kwargs)
        self.param_list['imagefolder'] = ['url', 'org_folder_name']
        self.class_to_idx: Dict[str, int] = \
            self.get_org_dataset('train').class_to_idx
        if self.num_classes is None:
            self.num_classes = len(self.class_to_idx)

    def initialize(self, verbose=True, **kwargs):
        file_path = self.download()
        uncompress(file_path=file_path.values(),
                   target_path=self.folder_path+self.name, verbose=verbose)
        os.rename(self.folder_path+self.name+'/{}/'.format(self.org_folder_name['train']),
                  self.folder_path+self.name+'/train/')
        if '/' in self.org_folder_name['train']:
            shutil.rmtree(self.folder_path+self.name + '/' +
                          self.org_folder_name['train'].split('/')[0])
        if self.valid_set:
            os.rename(self.folder_path+self.name+'/{}/'.format(self.org_folder_name['valid']),
                      self.folder_path+self.name+'/valid/')
            if '/' in self.org_folder_name['valid']:
                shutil.rmtree(self.folder_path+self.name + '/' +
                              self.org_folder_name['valid'].split('/')[0])

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default', **kwargs) -> datasets.ImageFolder:
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        return datasets.ImageFolder(root=self.folder_path+self.name+'/{}/'.format(mode),
                                    transform=transform)

    def download(self, url: Dict[str, str] = None, file_path: str = None,
                 folder_path: str = None, file_name: str = None, file_ext: str = 'zip',
                 verbose: bool = True):
        if url is None:
            url = self.url
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if file_name is None:
                file_name = {}
                file_path = {}
                file_name['train'] = self.name+'_train.'+file_ext
                file_path['train'] = folder_path+file_name['train']
                if self.valid_set:
                    file_name['valid'] = self.name+'_valid.'+file_ext
                    file_path['valid'] = folder_path+file_name['valid']
        for mode in file_path.keys():
            if not os.path.exists(file_path[mode]):
                if verbose:
                    print('Downloading Dataset %s ...' % self.name)
                urllib.request.urlretrieve(url[mode], file_path[mode])
                if verbose:
                    print('Dataset downloaded at file_path[mode]')
                    print()
            elif verbose:
                print('File Already Exists: ', file_path[mode])
                print()
        return file_path

    def sample(self, child_name: str = None, class_dict: dict = None, sample_num: int = None, output=True):
        if sample_num is None:
            assert class_dict is not None
            sample_num = len(class_dict)
        if child_name is None:
            child_name = self.name + '_sample%d' % sample_num
        src_path = self.folder_path+self.name+'/'
        mode_list = os.listdir(src_path)
        dst_path = self.folder_path+child_name+'/'

        if class_dict is None:
            assert sample_num is not None
            np.random.seed(env['seed'])
            idx_list = np.array(range(self.num_classes))
            np.random.shuffle(idx_list)
            idx_list = idx_list[:sample_num]
            class_list = np.array(os.listdir(src_path+mode_list[0]))[idx_list]
            class_dict = {}
            for class_name in class_list:
                class_dict[class_name] = [class_name]
        if output:
            class_list = tqdm(class_list)

        for src_mode in mode_list:
            if output:
                print(src_mode)
            assert src_mode in ['train', 'valid', 'test', 'val']
            dst_mode = 'valid' if src_mode == 'val' else src_mode
            if not os.path.exists(dst_path+dst_mode):
                os.makedirs(dst_path+dst_mode)
            for dst_class in class_dict.keys():
                class_list = class_dict[dst_class]
                for src_class in class_list:
                    shutil.copytree(src_path+src_mode+'/'+src_class,
                                    dst_path+dst_mode+'/'+dst_class)

    def split(self, ratio_dict={'train': 8, 'valid': 1, 'test': 1}, verbose=True):
        target_folder = self.folder_path+self.name+'/total/'
        train_folder = self.folder_path+self.name+'/train/'
        valid_folder = self.folder_path+self.name+'/valid/'
        test_folder = self.folder_path+self.name+'/test/'
        length = len(os.listdir(target_folder))
        ratio_sum = ratio_dict['train']+ratio_dict['valid']+ratio_dict['test']
        if verbose:
            print('Splitting Dataset...')
        for counter, _class in enumerate(os.listdir(target_folder)):
            if verbose:
                print(output_iter(counter, length))
            if not os.path.isdir(target_folder+_class):
                if verbose:
                    print('{red}Not a directory: {0}{1}{reset}'.format(
                        target_folder, _class, **ansi))
                continue
            if not os.path.exists(valid_folder+_class):
                os.makedirs(valid_folder+_class)
            if not os.path.exists(test_folder+_class):
                os.makedirs(test_folder+_class)
            if not os.path.exists(train_folder+_class):
                os.makedirs(train_folder+_class)
            seq = os.listdir(target_folder+_class)

            if verbose:
                print(_class, ' \t Image Number: ', len(seq))

            ratio_one = (len(seq)//ratio_sum)
            # valid
            for img in seq[:ratio_dict['valid']*ratio_one]:
                src = target_folder+_class+'/'+img
                dest = valid_folder+_class+'/'+img
                shutil.move(src, dest)
            # test
            for img in seq[ratio_dict['valid']*ratio_one:(ratio_dict['valid']+ratio_dict['test'])*ratio_one]:
                src = target_folder+_class+'/'+img
                dest = test_folder+_class+'/'+img
                shutil.move(src, dest)
            # train
            for img in seq[(ratio_dict['valid']+ratio_dict['test'])*ratio_one:]:
                src = target_folder+_class+'/'+img
                dest = train_folder+_class+'/'+img
                shutil.move(src, dest)
        shutil.rmtree(target_folder)
