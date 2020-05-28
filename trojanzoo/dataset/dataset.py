# -*- coding: utf-8 -*-

from trojanzoo.utils import to_tensor, to_numpy
from trojanzoo.utils.output import prints

import os
import torch
import numpy as np
from collections import OrderedDict

from trojanzoo.config import Config
env = Config.env


class Dataset:
    """An abstract class representing a Dataset.

    Args:
        name (string): Dataset Name. (need overwrite)
        data_type (string): Data type. (need overwrite)
        folder_path (string): dataset specific directory path,
                              defaults to ``env[data_dir]/[self.data_type]/[self.name]/data/``

    """

    def __init__(self, name='abstact', data_type='abstract', num_classes: int = None, test_set: bool = False,
                 batch_size: int = -128, num_workers: int = 4, train_num: int = 1024, loss_weights: bool = False,
                 folder_path: str = None, download: bool = False, **kwargs):

        self.name = name
        self.data_type = data_type
        self.num_classes = num_classes
        self.test_set = test_set
        self.param_list = OrderedDict()
        self.param_list['abstract'] = ['data_type', 'folder_path',
                                       'batch_size', 'num_classes', 'num_workers']
        # ----------------------------------------------------------------------------- #

        if batch_size < 0:
            batch_size = -batch_size * max(1, torch.cuda.device_count())
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.train_num = train_num
        # ----------------------------------------------------------------------------- #

        # Folder Path
        if folder_path is None:
            data_dir: str = env['data_dir']
            memory_dir: str = env['memory_dir']
            result_dir: str = env['result_dir']
            if memory_dir is not None:
                if not os.path.exists(memory_dir+data_type+'/'+name+'/data/'):
                    memory_dir = None
            if memory_dir is not None:
                folder_path = memory_dir+data_type+'/'+name+'/data/'
            else:
                folder_path = data_dir+data_type+'/'+name+'/data/'
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        # ----------------------------------------------------------------------------- #
        # Loss Weights
        if isinstance(loss_weights, bool):
            self.loss_weights = self.get_loss_weights() if loss_weights else None
        else:
            self.loss_weights = loss_weights

        # ----------------------------------------------------------------------------- #
        # Preset Loader
        self.loader = {}
        try:
            self.loader['train'] = self.get_dataloader(
                mode='train', batch_size=self.batch_size, full=True)
            self.loader['valid'] = self.get_dataloader(
                mode='valid', batch_size=self.batch_size, full=True)
            self.loader['valid2'] = self.get_dataloader(
                mode='valid', batch_size=self.batch_size, full=False)
            self.loader['test'] = self.get_dataloader(
                mode='test', batch_size=1)
        except Exception as e:
            if download:
                self.initialize()
                print('Dataset Initialized. You need to rerun the program.')
                raise SystemExit()
            else:
                raise e

    def initialize(self):
        raise NotImplementedError()

    def summary(self, indent: int = 0):
        prints('{:<10s} Parameters: '.format(self.name), indent=indent)
        d = self.__dict__
        for key, value in self.param_list.items():
            prints(key, indent=indent+10)
            prints({v: d[v] for v in value}, indent=indent+10)
            prints('-'*20, indent=indent+10)

    def get_transform(self, mode):
        pass

    @staticmethod
    def get_data(data, **kwargs):
        return data

    def get_full_dataset(self, mode, transform: object = None):
        return []

    def get_dataset(self, mode: str, full=True, **kwargs):
        if full:
            return self.get_full_dataset(mode)
        else:
            if mode == 'train':
                full_dataset = self.get_full_dataset(mode)
                indices = list(range(len(full_dataset)))
                np.random.seed(env['numpy_seed'])
                np.random.shuffle(indices)
                return torch.utils.data.Subset(full_dataset, indices[:self.train_num])
            else:
                return self.get_split_validset(mode, **kwargs)

    def get_dataloader(self, mode: str, full=False, batch_size: int = None, shuffle: bool = None, num_workers: int = None, **kwargs) -> torch.utils.data.dataloader:
        return []

    def get_split_validset(self, mode: str, valid_percent=0.6) -> torch.utils.data.dataloader:
        if self.test_set:
            return self.get_full_dataset(mode)
        full_dataset = self.get_full_dataset('valid')
        split = int(np.floor(valid_percent * len(full_dataset)))
        indices = list(range(len(full_dataset)))
        np.random.seed(env['numpy_seed'])
        np.random.shuffle(indices)
        if mode == 'test':
            return torch.utils.data.Subset(full_dataset, indices[split:])
        elif mode == 'valid':
            return torch.utils.data.Subset(full_dataset, indices[:split])
        else:
            raise ValueError(
                'argument \"mode\" value must be \"valid\" or \"test\"!')

    def get_loss_weights(self, file_path: str = None) -> torch.FloatTensor:
        if file_path is None:
            file_path = self.folder_path+'loss_weights.npy'
        if os.path.exists(file_path):
            loss_weights = to_tensor(np.load(file_path), dtype='float')
            return loss_weights
        else:
            print('Calculating Loss Weights')
            train_loader = self.get_dataloader('train', full=True)
            loss_weights = np.zeros(self.num_classes)
            for i, (X, Y) in enumerate(train_loader):
                Y = to_numpy(Y).tolist()
                for _class in range(self.num_classes):
                    loss_weights[_class] += Y.count(_class)
            loss_weights = loss_weights.sum() / loss_weights
            np.save(file_path, loss_weights)
            print('Loss Weights Saved at ', file_path)
            return to_tensor(loss_weights, dtype='float')

    def __str__(self):
        return self.summary()
