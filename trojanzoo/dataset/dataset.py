# -*- coding: utf-8 -*-

from trojanzoo.utils import to_tensor, to_list
from trojanzoo.utils.output import prints, Indent_Redirect

import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from typing import Union, List, Tuple, Dict

from trojanzoo.utils import Config
env = Config.env

redirect = Indent_Redirect(buffer=True, indent=0)


class Dataset:
    """An abstract class representing a Dataset.

    Args:
        name (string): Dataset Name. (need override)
        data_type (string): Data type. (need override)
        folder_path (string): dataset specific directory path,
                              defaults to ``env[data_dir]/[self.data_type]/[self.name]/data/``

    """
    name: str = 'abstact'
    data_type: str = 'abstract'
    num_classes: int = None
    label_names: List[int] = []
    valid_set: bool = True

    def __init__(self, batch_size: int = -128, folder_path: str = None, download: bool = False,
                 split_ratio: float = 0.8, train_sample: int = 1024, test_ratio: float = 0.3,
                 num_workers: int = 0, loss_weights: bool = False, **kwargs):

        self.param_list: Dict[str, List[str]] = OrderedDict()
        self.param_list['abstract'] = ['data_type', 'folder_path', 'label_names',
                                       'batch_size', 'num_classes', 'num_workers']
        if batch_size < 0:
            batch_size = -batch_size * max(1, torch.cuda.device_count())
        self.batch_size = batch_size

        self.split_ratio = split_ratio
        self.train_sample = train_sample
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        # ----------------------------------------------------------------------------- #

        # Folder Path
        if folder_path is None:
            data_dir: str = env['data_dir']
            memory_dir: str = env['memory_dir']
            result_dir: str = env['result_dir']
            if memory_dir:
                if not os.path.exists(memory_dir + self.data_type + '/' + self.name + '/data/'):
                    memory_dir = None
            _dir = memory_dir if memory_dir else data_dir
            folder_path = _dir + self.data_type + '/' + self.name + '/data/'
        self.folder_path: str = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        # ----------------------------------------------------------------------------- #
        if download:
            if not self.check_files():
                self.initialize()
        # Preset Loader
        self.loader: Dict[str, torch.utils.data.DataLoader] = {}
        self.loader['train'] = self.get_dataloader(
            mode='train', batch_size=self.batch_size, full=True)
        self.loader['train2'] = self.get_dataloader(
            mode='train', batch_size=self.batch_size, full=False)
        self.loader['valid'] = self.get_dataloader(
            mode='valid', batch_size=self.batch_size, full=True)
        self.loader['valid2'] = self.get_dataloader(
            mode='valid', batch_size=self.batch_size, full=False)
        self.loader['test'] = self.get_dataloader(
            mode='test', batch_size=1)
        # ----------------------------------------------------------------------------- #
        # Loss Weights
        self.loss_weights: torch.FloatTensor = None
        if isinstance(loss_weights, bool):
            self.loss_weights = self.get_loss_weights() if loss_weights else None
        else:
            self.loss_weights = loss_weights

    def check_files(self) -> bool:
        try:
            dataset = self.get_org_dataset(mode='train')
        except Exception:
            return False
        else:
            return True

    def initialize(self, verbose=True):
        raise NotImplementedError()

    def summary(self, indent: int = 0):
        prints('{:<10s} Parameters: '.format(self.name), indent=indent)
        d = self.__dict__
        for key, value in self.param_list.items():
            prints(key, indent=indent + 10)
            prints({v: getattr(self, v) for v in value}, indent=indent + 10)
            prints('-' * 20, indent=indent + 10)

    @classmethod
    def get_transform(cls, mode: str) -> object:
        pass

    @staticmethod
    def get_data(data: Tuple[torch.Tensor], **kwargs) -> (torch.Tensor, torch.LongTensor):
        return data

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default',
                        **kwargs) -> torch.utils.data.Dataset:
        pass

    def get_full_dataset(self, mode: str, **kwargs) -> torch.utils.data.Dataset:
        try:
            if self.valid_set:
                return self.get_org_dataset(mode, **kwargs)
            else:
                dataset = self.get_org_dataset(mode='train', **kwargs)
                subset = {}
                subset['train'], subset['valid'] = self.split_set(
                    dataset, percent=self.split_ratio)
                return subset[mode]
        except RuntimeError as e:
            print(self.folder_path)
            raise e

    def get_dataset(self, mode: str, full: bool = True, classes: List[int] = None, **kwargs) -> torch.utils.data.Dataset:
        if full and mode != 'test':
            dataset = self.get_full_dataset(mode=mode, **kwargs)
        elif mode == 'train':
            fullset = self.get_full_dataset(mode='train', **kwargs)
            dataset, _ = self.split_set(fullset, length=self.train_sample)
        else:
            fullset = self.get_full_dataset(mode='valid', **kwargs)
            subset: Dict[str, torch.utils.data.Subset] = {}
            subset['test'], subset['valid'] = self.split_set(
                fullset, percent=self.test_ratio)
            dataset = subset[mode]
        if classes:
            dataset = self.get_class_set(dataset, classes=classes, **kwargs)
        return dataset

    def get_class_set(self, dataset: torch.utils.data.Dataset, classes: List[int], **kwargs):
        indices = np.arange(len(dataset))
        if isinstance(dataset, torch.utils.data.Subset):
            idx = np.array(dataset.indices)
            indices = idx[indices]
            dataset = dataset.dataset
        idx_bool = np.isin(dataset.targets, classes)
        idx = np.arange(len(dataset))[idx_bool]
        idx = np.intersect1d(idx, indices)
        return torch.utils.data.Subset(dataset, idx)

    def get_dataloader(self, mode: str, batch_size: int = None, shuffle: bool = None,
                       num_workers: int = None, pin_memory=True, **kwargs) -> torch.utils.data.dataloader:
        pass

    @classmethod
    def split_set(cls, dataset: Union[torch.utils.data.Dataset, torch.utils.data.Subset],
                  length: int = None, percent=None) -> (torch.utils.data.Subset, torch.utils.data.Subset):
        assert (length is None) != (percent is None)  # XOR check
        if length is None:
            length = int(len(dataset) * percent)
        indices = list(range(len(dataset)))
        np.random.seed(env['seed'])
        np.random.shuffle(indices)
        if isinstance(dataset, torch.utils.data.Subset):
            idx = torch.as_tensor(dataset.indices)
            indices = idx[indices]
            dataset = dataset.dataset
        subset1 = torch.utils.data.Subset(dataset, indices[:length])
        subset2 = torch.utils.data.Subset(dataset, indices[length:])
        return subset1, subset2

    def get_loss_weights(self, file_path: str = None, verbose: bool = True) -> torch.FloatTensor:
        if file_path is None:
            file_path = self.folder_path + 'loss_weights.npy'
        if os.path.exists(file_path):
            loss_weights = to_tensor(np.load(file_path), dtype='float')
            return loss_weights
        else:
            if verbose:
                print('Calculating Loss Weights')
            loss_weights = np.zeros(self.num_classes)
            for X, Y in self.loader['train']:
                Y = to_list(Y)
                for _class in range(self.num_classes):
                    loss_weights[_class] += Y.count(_class)
            loss_weights = loss_weights.sum() / loss_weights
            np.save(file_path, loss_weights)
            if verbose:
                print('Loss Weights Saved at ', file_path)
            return to_tensor(loss_weights, dtype='float')

    def __str__(self) -> str:
        sys.stdout = redirect
        self.summary()
        _str = redirect.buffer
        redirect.reset()
        return _str
