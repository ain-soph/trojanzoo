# -*- coding: utf-8 -*-

from trojanzoo.utils import to_tensor, to_numpy
from trojanzoo.utils.output import prints

import os
import torch
import numpy as np

from trojanzoo.config import Config
config=Config.config


class Dataset(object):
    """An abstract class representing a Dataset.

    Args:
        name (string): Dataset Name. (need overwrite)
        data_type (string): Data type. (need overwrite)
        folder_path (string): dataset specific directory path,
                              defaults to ``[data_dir]/[data_type]/[name]/data/``

    """

    def __init__(self, name='abstact', data_type='abstract',
                 folder_path: str = None,
                 batch_size: int = 128, num_classes: int = None, test_set: bool = False, loss_weights: bool = False,
                 train_num: int = 1024, num_workers: int = 4,
                 default_model: str = 'default', **kwargs):

        self.name = name
        self.param_dict = {}
        self.param_dict['abstract'] = {'path': ['data_dir', 'result_dir', 'memory_dir', 'folder_path'],
                                       'param': ['batch_size', 'num_classes', 'data_type']}

        self.data_type = data_type
        self.data_dir = config['general']['path']['data_dir'],
        self.result_dir = result_dir

        if memory_dir is not None:
            if not os.path.exists(memory_dir+self.data_type+'/'+self.name+'/data/'):
                memory_dir = None
        self.folder_path = folder_path
        if self.folder_path is None:
            if memory_dir is not None:
                self.folder_path = memory_dir+self.data_type+'/'+self.name+'/data/'
            else:
                self.folder_path = self.data_dir+self.data_type+'/'+self.name+'/data/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.train_num = train_num
        self.num_workers = num_workers

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.test_set = test_set
        if isinstance(loss_weights, bool):
            if loss_weights:
                self.loss_weights = self.get_loss_weights()
            else:
                self.loss_weights = None
        else:
            self.loss_weights = loss_weights

        self.default_model = default_model

        self.loader = {}
        self.loader['train'] = self.get_dataloader(
            mode='train', batch_size=self.batch_size, full=True)
        self.loader['valid'] = self.get_dataloader(
            mode='valid', batch_size=self.batch_size, full=True)
        self.loader['valid2'] = self.get_dataloader(
            mode='valid', batch_size=self.batch_size, full=False)
        self.loader['test'] = self.get_dataloader(mode='test', batch_size=1)

    def initialize(self):
        pass

    def summary(self, indent: int = 0):
        prints('{:<10s} Parameters: '.format(self.name), indent=indent)
        d = self.__dict__
        prints(indent=indent)
        print()

    def get_transform(self, mode):
        pass

    @staticmethod
    def get_data(data):
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
                np.random.seed(self.numpy_seed)
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
        np.random.seed(self.numpy_seed)
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

    # def get_model(self, model_name=None, *args, **kwargs):
    #     if model_name is None:
    #         model_name = self.default_model
    #     get_model(model_name, *args, **kwargs)
