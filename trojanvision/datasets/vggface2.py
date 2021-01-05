#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imagefolder import ImageFolder
from trojanzoo.utils.param import Module

import torchvision.transforms as transforms
import numpy as np
import os
import sys
import requests
import getpass


class VGGface2(ImageFolder):

    name: str = 'vggface2'
    n_dim = (224, 224)
    num_classes = 8631
    valid_set: bool = False
    org_folder_name = {'train': 'train'}

    @staticmethod
    def get_transform(mode: str) -> transforms.Compose:
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor()])
        return transform

    def initialize_npz(self, mode_list: list[str] = ['train', 'valid'],
                       transform: transforms.Compose = transforms.Compose([transforms.Resize((256, 256)),
                                                                           transforms.Lambda(lambda x: np.array(x))]),
                       **kwargs):
        super().initialize_npz(mode_list=mode_list, transform=transform, **kwargs)

    def download(self, mode: str = 'train', file_path=None, folder_path=None, file_name=None, file_ext='tar.gz', **kwargs):
        assert mode == 'train'
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if file_name is None:
                file_name = f'{self.name}_{mode}.{file_ext}'
                file_path = folder_path + file_name
        if os.path.exists(file_path['train']):
            print('File Already Exists: ', file_path)
            return file_path

        LOGIN_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/login/"
        FILE_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz"

        print('Please enter your VGG Face 2 credentials:')
        user_string = input('    User: ')
        password_string = getpass.getpass(prompt='    Password: ')

        payload = {
            'username': user_string,
            'password': password_string
        }

        session = requests.session()
        r = session.get(LOGIN_URL)

        if 'csrftoken' in session.cookies:
            csrftoken = session.cookies['csrftoken']
        elif 'csrf' in session.cookies:
            csrftoken = session.cookies['csrf']
        else:
            raise ValueError("Unable to locate CSRF token.")

        payload['csrfmiddlewaretoken'] = csrftoken

        r = session.post(LOGIN_URL, data=payload)

        # filename = FILE_URL.split('=')[-1]

        with open(file_path['train'], "wb") as f:
            print(f"Downloading file: `{file_path['train']}`")
            r = session.get(FILE_URL, data=payload, stream=True)
            bytes_written = 0
            for data in r.iter_content(chunk_size=4096):
                f.write(data)
                bytes_written += len(data)
                MiB = bytes_written / (1024 * 1024)
                sys.stdout.write(f"\r{MiB:0.2f} MiB downloaded...")
                sys.stdout.flush()

        print("\nDone.")
        return file_path


class Sample_VGGface2(VGGface2):

    name: str = 'sample_vggface2'
    num_classes = 20
    url = {}
    org_folder_name = {}

    def initialize_folder(self):
        _dict = Module(self.__dict__)
        _dict.__delattr__('folder_path')
        vggface2 = VGGface2(**_dict)
        vggface2.sample(child_name=self.name, sample_num=self.num_classes)
