#!/usr/bin/env python3

from trojanzoo.configs import config, Config
from trojanzoo.environ import env
from trojanzoo.utils import get_name, to_tensor
from trojanzoo.utils.data import dataset_to_list, split_dataset, get_class_subset
from trojanzoo.utils.output import ansi, prints, Indent_Redirect

import torch
import numpy as np
import os
import sys

from typing import TYPE_CHECKING
from typing import Callable, Union    # TODO: python 3.10
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.utils.data


redirect = Indent_Redirect(buffer=True, indent=0)


class Dataset:
    """An abstract class representing a Dataset.

    Args:
        name (string): Dataset Name. (need override)
        data_type (string): Data type (e.g., 'image'). (need override)
        folder_path (string): directory path to store dataset.

    """
    name: str = None
    data_type: str = None
    num_classes: int = None
    label_names: list[int] = None
    valid_set = True

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('-d', '--dataset', dest='dataset_name', type=str,
                           help='dataset name (lowercase).')
        group.add_argument('--batch_size', dest='batch_size', type=int,
                           help='batch size (negative number means batch_size for each gpu).')
        group.add_argument('--valid_batch_size', dest='valid_batch_size', type=int,
                           help='valid batch size.')
        group.add_argument('--test_batch_size', dest='test_batch_size', type=int,
                           help='test batch size.')
        group.add_argument('--num_workers', dest='num_workers', type=int,
                           help='num_workers passed to torch.utils.data.DataLoader for training set, defaults to 4.')
        group.add_argument('--download', dest='download', action='store_true',
                           help='download dataset if not exist by calling dataset.initialize()')
        group.add_argument('--data_dir', dest='data_dir',
                           help='directory to contain datasets')
        return group

    def __init__(self, batch_size: int = None, folder_path: str = None, download: bool = False,
                 split_ratio: float = 0.8, train_sample: int = 1024, test_ratio: float = 0.3,
                 num_workers: int = 4, loss_weights: Union[bool, np.ndarray] = False,
                 valid_batch_size: int = 100, test_batch_size: int = 1, **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['dataset'] = ['data_type', 'folder_path', 'label_names',
                                      'batch_size', 'num_classes', 'num_workers',
                                      'valid_batch_size', 'test_batch_size']
        self.__batch_size: int = 0
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.split_ratio = split_ratio
        self.train_sample = train_sample
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        # ----------------------------------------------------------------------------- #

        self.folder_path = folder_path
        if folder_path is not None:
            self.folder_path = os.path.normpath(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        # ----------------------------------------------------------------------------- #
        if download and not self.check_files():
            self.initialize()
        # Preset Loader
        self.loader: dict[str, torch.utils.data.DataLoader] = {}
        self.loader['train'] = self.get_dataloader(mode='train')
        self.loader['train2'] = self.get_dataloader(mode='train', full=False)
        self.loader['valid'] = self.get_dataloader(mode='valid')
        self.loader['valid2'] = self.get_dataloader(mode='valid', full=False)
        self.loader['test'] = self.get_dataloader(mode='test')
        # ----------------------------------------------------------------------------- #
        # Loss Weights
        self.loss_weights: np.ndarray = loss_weights
        if isinstance(loss_weights, bool):
            self.loss_weights = self.get_loss_weights() if loss_weights else None

    def check_files(self, transform: str = None, **kwargs) -> bool:
        try:
            self.get_org_dataset(mode='train', transform=transform, **kwargs)
            if self.valid_set:
                self.get_org_dataset(mode='valid', transform=transform, **kwargs)
        except Exception:
            return False
        return True

    def initialize(self, verbose: bool = None):
        raise NotImplementedError()

    def summary(self, indent: int = 0):
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(self.name, **ansi), indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key, value in self.param_list.items():
            if value:
                prints('{green}{0:<20s}{reset}'.format(key, **ansi), indent=indent + 10)
                prints({v: getattr(self, v) for v in value}, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)

    def get_transform(self, mode: str) -> Callable:
        raise NotImplementedError()

    @staticmethod
    def get_data(data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return data

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default',
                        **kwargs) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def get_full_dataset(self, mode: str, transform='default', seed: int = None, **kwargs) -> torch.utils.data.Dataset:
        try:
            if self.valid_set:
                return self.get_org_dataset(mode=mode, transform=transform, **kwargs)
            else:
                dataset = self.get_org_dataset(mode='train', transform=transform, **kwargs)
                subset = {}
                subset['train'], subset['valid'] = self.split_dataset(
                    dataset, percent=self.split_ratio, seed=seed)
                return subset[mode]
        except RuntimeError as e:
            print(f'{self.folder_path=}')
            raise e

    def get_dataset(self, mode: str = None, full: bool = True, dataset: torch.utils.data.Dataset = None,
                    classes: list[int] = None, seed: int = None, **kwargs) -> torch.utils.data.Dataset:
        if dataset is None:
            if full and mode != 'test':
                dataset = self.get_full_dataset(mode=mode, **kwargs)
            elif mode == 'train':
                fullset = self.get_full_dataset(mode='train', **kwargs)
                dataset, _ = self.split_dataset(fullset, length=self.train_sample, seed=seed)
            else:
                fullset = self.get_full_dataset(mode='valid', **kwargs)
                subset: dict[str, torch.utils.data.Subset] = {}
                subset['test'], subset['valid'] = self.split_dataset(
                    fullset, percent=self.test_ratio, seed=seed)
                dataset = subset[mode]
        if classes is not None:
            dataset = self.get_class_set(dataset=dataset, classes=classes)
        return dataset

    @staticmethod
    def get_class_subset(dataset: torch.utils.data.Dataset,
                         classes: list[int]) -> torch.utils.data.Subset:
        return get_class_subset(dataset, classes)

    def get_dataloader(self, mode: str = None, dataset: torch.utils.data.Dataset = None,
                       batch_size: int = None, shuffle: bool = None,
                       num_workers: int = None, pin_memory=True, drop_last=False, **kwargs) -> torch.utils.data.DataLoader:
        if batch_size is None:
            # TODO: python 3.10 match
            if mode == 'train':
                batch_size = self.batch_size
            elif mode == 'valid':
                batch_size = self.valid_batch_size
            else:
                assert mode == 'test'
                batch_size = self.test_batch_size
        if shuffle is None:
            shuffle = True if mode == 'train' else False
        num_workers = num_workers if num_workers is not None else self.num_workers
        dataset = self.get_dataset(mode=mode, dataset=dataset, **kwargs)
        if env['num_gpus'] == 0:
            pin_memory = False
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    @staticmethod
    def split_dataset(dataset: Union[torch.utils.data.Dataset, torch.utils.data.Subset],
                  length: int = None, percent=None, seed: int = None
                  ) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        seed = env['seed'] if seed is None else seed
        return split_dataset(dataset, length, percent, seed)

    def get_loss_weights(self, file_path: str = None, verbose: bool = None) -> np.ndarray:
        file_path = file_path if file_path is not None else os.path.join(self.folder_path, 'loss_weights.npy')
        if os.path.exists(file_path):
            loss_weights = to_tensor(np.load(file_path), dtype='float')
            return loss_weights
        else:
            if verbose:
                print('Calculating Loss Weights')
            dataset = self.get_full_dataset('train', transform=None)
            _, targets = dataset_to_list(dataset, label_only=True)
            loss_weights = np.bincount(targets)     # TODO: linting problem
            assert len(loss_weights) == self.num_classes
            loss_weights: np.ndarray = loss_weights.sum() / loss_weights     # TODO: linting problem
            np.save(file_path, loss_weights)
            print('Loss Weights Saved at ', file_path)
            return loss_weights

    def __str__(self) -> str:
        sys.stdout = redirect
        self.summary()
        _str = redirect.buffer
        redirect.reset()
        return _str

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.__batch_size = value if value >= 0 else -value * max(1, torch.cuda.device_count())


def add_argument(parser: argparse.ArgumentParser, dataset_name: str = None, dataset: Union[str, Dataset] = None,
                 config: Config = config, class_dict: dict[str, type[Dataset]] = {}) -> argparse._ArgumentGroup:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None else config.get_full_config()['dataset']['default_dataset']
    group = parser.add_argument_group('{yellow}dataset{reset}'.format(**ansi), description=dataset_name)
    try:
        DatasetType = class_dict[dataset_name]
    except KeyError as e:
        print(f'{dataset_name} not in \n{list(class_dict.keys())}')
        raise e
    return DatasetType.add_argument(group)


def create(dataset_name: str = None, dataset: str = None, folder_path: str = None,
           config: Config = config, class_dict: dict[str, type[Dataset]] = {}, **kwargs) -> Dataset:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None else config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['dataset']._update(kwargs)
    try:
        DatasetType = class_dict[dataset_name]
    except KeyError as e:
        print(f'{dataset_name} not in \n{list(class_dict.keys())}')
        raise e
    folder_path = folder_path if folder_path is not None \
        else os.path.join(result['data_dir'], DatasetType.data_type, DatasetType.name)
    return DatasetType(folder_path=folder_path, **result)
