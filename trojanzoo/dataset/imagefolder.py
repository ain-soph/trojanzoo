# -*- coding: utf-8 -*-

from .imageset import ImageSet
from trojanzoo.utils.os import uncompress

import os
import shutil
import numpy as np
from tqdm import tqdm
import urllib.request
import torchvision.datasets as datasets

from trojanzoo.config import Config
env = Config.env


class ImageFolder(ImageSet):
    """docstring for dataset"""

    def __init__(self, name='imagefolder', **kwargs):
        super(ImageFolder, self).__init__(name=name, **kwargs)
        if self.num_classes is None:
            self.num_classes = len(os.listdir(
                self.folder_path+self.name+'/train/'))

    def initialize(self, valid=False, output=True, **kwargs):
        file_path = self.download()
        uncompress(file_path=file_path,
                   target_path=self.folder_path+self.name, output=output)
        if valid:
            os.rename(self.folder_path+self.name+'/%s/' % getattr(self, 'org_folder_name')['train'],
                      self.folder_path+self.name+'/train/')
            os.rename(self.folder_path+self.name+'/%s/' % getattr(self, 'org_folder_name')['valid'],
                      self.folder_path+self.name+'/valid/')
        else:
            os.rename(self.folder_path+self.name+'/%s/' % getattr(self, 'org_folder_name')['train'],
                      self.folder_path+self.name+'/total/')
            self.split(output=output, **kwargs)

    def get_full_dataset(self, mode):
        return datasets.ImageFolder(root=self.folder_path+self.name+'/'+mode+'/', transform=self.get_transform(mode))

    def split(self, ratio_dict={'train': 8, 'valid': 1, 'test': 1}, output=True):
        target_folder = self.folder_path+self.name+'/total/'
        train_folder = self.folder_path+self.name+'/train/'
        valid_folder = self.folder_path+self.name+'/valid/'
        test_folder = self.folder_path+self.name+'/test/'
        print('Splitting Dataset...')
        ratio_sum = ratio_dict['train']+ratio_dict['valid']+ratio_dict['test']

        length = len(os.listdir(target_folder))
        for counter, _class in enumerate(os.listdir(target_folder)):
            if not os.path.isdir(target_folder+_class):
                print(_class+' is not a directory...')
                continue
            if not os.path.exists(valid_folder+_class):
                os.makedirs(valid_folder+_class)
            if not os.path.exists(test_folder+_class):
                os.makedirs(test_folder+_class)
            if not os.path.exists(train_folder+_class):
                os.makedirs(train_folder+_class)
            seq = os.listdir(target_folder+_class)

            if output:
                counter += 1
                print('[%d/%d]' % (counter, length), _class,
                      ' \t Image Number: ', len(seq))

            # valid
            for img in seq[:ratio_dict['valid']*(len(seq)//ratio_sum)]:
                src = target_folder+_class+'/'+img
                dest = valid_folder+_class+'/'+img
                shutil.move(src, dest)
            # test
            for img in seq[ratio_dict['valid']*(len(seq)//ratio_sum):(ratio_dict['valid']+ratio_dict['test'])*(len(seq)//ratio_sum)]:
                src = target_folder+_class+'/'+img
                dest = test_folder+_class+'/'+img
                shutil.move(src, dest)
            # train
            for img in seq[(ratio_dict['valid']+ratio_dict['test'])*(len(seq)//ratio_sum):]:
                src = target_folder+_class+'/'+img
                dest = train_folder+_class+'/'+img
                shutil.move(src, dest)
        shutil.rmtree(target_folder)

    def download(self, url: str = None, file_path: str = None, folder_path: str = None, file_name: str = None, file_ext='zip', valid=False, output=True):
        if url is None:
            url = getattr(self, 'url')
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if file_name is None:
                if valid:
                    file_name = {'train': self.name+'_train.'+file_ext,
                                 'valid': self.name+'_valid.'+file_ext}
                    file_path = {'train': folder_path+file_name['train'],
                                 'valid': folder_path+file_name['valid']}
                else:
                    file_name = {'train': self.name+'_train.'+file_ext}
                    file_path = {'train': folder_path+file_name['train']}
        for mode in file_path.keys():
            if not os.path.exists(file_path[mode]):
                print('Downloading Dataset %s ...' % self.name)
                urllib.request.urlretrieve(url, file_path[mode])
                print('Dataset downloaded at file_path[mode]')
                print()
            else:
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
            np.random.seed(env['numpy_seed'])
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
