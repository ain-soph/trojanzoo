#!/usr/bin/env python3

from trojanzoo.configs import config
from trojanzoo.environ import env
from trojanzoo.utils.data import get_class_subset, split_dataset
from trojanzoo.utils.module import BasicObject, get_name
from trojanzoo.utils.output import ansi

import torch
import numpy as np
import functools
import os
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from typing import Iterable
from trojanzoo.configs import Config
import argparse    # TODO: python 3.10
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data


class Dataset(ABC, BasicObject):
    r"""
    | An abstract class representing a dataset.
    | It inherits :class:`trojanzoo.utils.module.BasicObject`.

    Note:
        This is the implementation of dataset.
        For users, please use :func:`create` instead, which is more user-friendly.

    Args:
        batch_size (int): Batch size of training set
            (negative number means batch size for each gpu).
        valid_batch_size (int): Batch size of validation set.
            Defaults to ``100``.
        folder_path (str): Folder path to store dataset.
            Defaults to ``None``.

            Note:
                :attr:`folder_path` is usually
                ``'{data_dir}/{data_type}/{name}'``,
                which is claimed as the default value of :func:`create()`.
        download (bool): Download dataset if not exist. Defaults to ``False``.
        split_ratio (float):
            | Split training set for training and validation
              if :attr:`valid_set` is ``False``.
            | The ratio stands for
              :math:`\frac{\text{\# training\ subset}}{\text{\# total\ training\ set}}`.
            | Defaults to ``0.8``.
        num_workers (int): Used in :meth:`get_dataloader()`.
            Defaults to ``4``.
        loss_weights (bool | np.ndarray | torch.Tensor):
            | The loss weights w.r.t. each class.
            | if :any:`numpy.ndarray` or :any:`torch.Tensor`,
              directly set as :attr:`loss_weights` (cpu tensor).
            | if ``True``, set :attr:`loss_weights` as :meth:`get_loss_weights()`;
            | if ``False``, set :attr:`loss_weights` as ``None``.
        **kwargs: Any keyword argument (unused).

    Attributes:
        name (str): Dataset Name. (need overriding)
        loader(dict[str, ~torch.utils.data.DataLoader]):
            | Preset dataloader for users at dataset initialization.
            | It contains ``'train'`` and ``'valid'`` loaders.
        batch_size (int): Batch size of training set (always positive).
        valid_batch_size (int): Batch size of validation set.
            Defaults to ``100``.
        num_classes (int): Number of classes. (need overriding)
        folder_path (str): Folder path to store dataset.
            Defaults to ``None``.

        data_type (str): Data type (e.g., ``'image'``). (need overriding)
        label_names (list[int]): Number of classes. (optional)
        valid_set (bool): Whether having a native validation set.
            Defaults to ``True``.
        split_ratio (float):
            | Split training set for training and validation
              if :attr:`valid_set` is ``False``.
            | The ratio stands for
              :math:`\frac{\text{\# training\ subset}}{\text{\# total\ training\ set}}`.
            | Defaults to ``0.8``.
        loss_weights (torch.Tensor | None): The loss weights w.r.t. each class.
        num_workers (int): Used in :meth:`get_dataloader()`.
            Defaults to ``4``.
        collate_fn (~collections.abc.Callable | None):
            Used in :meth:`get_dataloader()`.
            Defaults to ``None``.
    """
    name = 'dataset'
    data_type: str = None
    num_classes: int = None
    label_names: list[int] = None
    valid_set = True

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        r"""Add dataset arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete dataset class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
        group.add_argument('-d', '--dataset', dest='dataset_name',
                           help='dataset name (lowercase)')
        group.add_argument('--batch_size', type=int,
                           help='batch size (negative number means '
                           'batch_size for each gpu)')
        group.add_argument('--valid_batch_size', type=int,
                           help='valid batch size')
        group.add_argument('--num_workers', type=int,
                           help='num_workers passed to '
                           'torch.utils.data.DataLoader (default: 4)')
        group.add_argument('--download', action='store_true',
                           help='download dataset if not exist by calling '
                           'self.initialize()')
        group.add_argument('--data_dir', help='directory to contain datasets')
        return group

    def __init__(self, batch_size: int = None,
                 valid_batch_size: int = 100,
                 folder_path: str = None, download: bool = False,
                 split_ratio: float = 0.8, num_workers: int = 4,
                 loss_weights: bool | np.ndarray | torch.Tensor = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['dataset'] = ['num_classes', 'batch_size', 'valid_batch_size',
                                      'folder_path', 'num_workers', ]
        if not self.valid_set:
            self.param_list['dataset'].append('split_ratio')
        self.__batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.collate_fn: Callable[[Iterable[torch.Tensor]], Iterable[torch.Tensor]] = None
        # ----------------------------------------------- #

        if folder_path is not None:
            folder_path = os.path.normpath(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        self.folder_path = folder_path
        # ----------------------------------------------- #
        if download and not self.check_files():
            self.initialize()
        # Preset Loader
        try:
            self.loader: dict[str, torch.utils.data.DataLoader] = {}
            self.loader['train'] = self.get_dataloader(mode='train')
            self.loader['valid'] = self.get_dataloader(mode='valid')
        except Exception:
            print(f'Dataset Folder Path: {self.folder_path}')
            raise
        # ----------------------------------------------- #
        # Loss Weights
        match loss_weights:
            case bool():
                # TODO: issue 5 pylance
                loss_weights = self.get_loss_weights() if loss_weights else None
            case np.ndarray():
                loss_weights = torch.from_numpy(loss_weights).to(device=env['device'])
            case torch.Tensor():
                loss_weights = loss_weights.to(device=env['device'])
            case _:
                raise TypeError(type(loss_weights))
        self.loss_weights: None | torch.Tensor = loss_weights

    @functools.cached_property
    def batch_size(self):
        return self.__batch_size if self.__batch_size >= 0 else \
            -self.__batch_size * max(1, env['num_gpus'])

    # TODO: should it be abstractmethod?
    def initialize(self, *args, **kwargs):
        r"""Initialize the dataset (download and extract) if it's not prepared yet
        (need overriding).
        """
        raise NotImplementedError()

    def check_files(self, **kwargs) -> bool:
        r"""Check if the dataset files are prepared.

        Args:
            **kwargs: Keyword arguments passed to :meth:`get_org_dataset`.

        Returns:
            bool: Whether the dataset files are prepared.
        """
        try:
            self.get_org_dataset(mode='train', transform=None, **kwargs)
            if self.valid_set:
                self.get_org_dataset(
                    mode='valid', transform=None, **kwargs)
            return True
        except Exception:
            return False

    @abstractmethod
    def get_transform(self, mode: str) -> Callable:
        r"""Get dataset transform for mode.

        Args:
            mode (str): Dataset mode (e.g., ``'train'`` or ``'valid'``).

        Returns:
            ~collections.abc.Callable: A callable transform.
        """
        ...

    def get_data(self, data, **kwargs):
        r"""Process data. Defaults to directly return :attr:`data`.

        Args:
            data (Any): Unprocessed data.
            **kwargs: Keyword arguments to process data.

        Returns:
            Any: Processed data.
        """
        return data

    def get_org_dataset(self, mode: str,
                        **kwargs) -> torch.utils.data.Dataset:
        r"""Get original dataset that is not splitted.

        Note:
            This is a wrapper and the specific implementation
            is in :meth:`_get_org_dataset`, which needs overriding.

        Args:
            mode (str): Dataset mode (e.g., ``'train'`` or ``'valid'``).
            transform (~collections.abc.Callable):
                The transform applied on dataset.
                Defaults to :meth:`get_transform()`.
            **kwargs: Keyword arguments passed to :meth:`_get_org_dataset`.

        Returns:
            torch.utils.data.Dataset: The original dataset.

        See Also:
            :meth:`get_dataset`
        """
        if 'transform' not in kwargs.keys():
            kwargs['transform'] = self.get_transform(mode=mode)
        return self._get_org_dataset(mode=mode, **kwargs)

    @abstractmethod
    def _get_org_dataset(self, mode: str, transform: object = None,
                         **kwargs) -> torch.utils.data.Dataset:
        ...

    def get_dataset(self, mode: str = None, seed: int = None,
                    class_list: None | int | list[int] = None,
                    **kwargs):
        r"""Get dataset. Call :meth:`split_dataset` to split the training set
        if :attr:`valid_set` is ``False``.

        Args:
            mode (str): Dataset mode (e.g., ``'train'`` or ``'valid'``).
            seed (int): The random seed to split dataset
                using :any:`numpy.random.shuffle`.
                Defaults to ``env['data_seed']``.
            class_list (int | list[int]):
                The class list to pick. Defaults to ``None``.
            **kwargs: Keyword arguments passed to :meth:`get_org_dataset`.

        Returns:
            torch.utils.data.Dataset: The original dataset.
        """
        try:
            if self.valid_set or mode not in ['train', 'valid']:
                dataset = self.get_org_dataset(mode=mode, **kwargs)
            else:
                dataset = self.get_org_dataset(mode='train', **kwargs)
                subset: dict[str, torch.utils.data.Subset] = {}
                subset['train'], subset['valid'] = self.split_dataset(
                    dataset, percent=self.split_ratio, seed=seed)
                dataset = subset[mode]
        except RuntimeError:
            print(f'{self.folder_path=}')
            raise
        if class_list is not None:
            dataset = self.get_class_subset(dataset=dataset, class_list=class_list)
        return dataset

    @staticmethod
    def split_dataset(dataset: torch.utils.data.Dataset | torch.utils.data.Subset,
                      length: int = None, percent: float = None,
                      shuffle: bool = True, seed: int = None):
        r"""Split a dataset into two subsets.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to split.
            length (int): The length of the first subset.
                This argument cannot be used together with :attr:`percent`.
                If ``None``, use :attr:`percent` to calculate length instead.
                Defaults to ``None``.
            percent (float): The split ratio for the first subset.
                This argument cannot be used together with :attr:`length`.
                ``length = percent * len(dataset)``.
                Defaults to ``None``.
            shuffle (bool): Whether to shuffle the dataset.
                Defaults to ``True``.
            seed (bool): The random seed to split dataset
                using :any:`numpy.random.shuffle`.
                Defaults to ``env['data_seed']``.

        Returns:
            (torch.utils.data.Subset, torch.utils.data.Subset):
                The two splitted subsets.

        :Example:
            >>> from trojanzoo.utils.data import TensorListDataset
            >>> from trojanzoo.datasets import Dataset
            >>> import torch
            >>>
            >>> data = torch.ones(11, 3, 32, 32)
            >>> targets = list(range(11))
            >>> dataset = TensorListDataset(data, targets)
            >>> set1, set2 = Dataset.split_dataset(dataset, length=3)
            >>> len(set1), len(set2)
            (3, 8)
            >>> set3, set4 = split_dataset(dataset, percent=0.5)
            >>> len(set3), len(set4)
            (5, 6)

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.data.split_dataset`.
            The difference is that this method will set :attr:`seed`
            as ``env['data_seed']`` when it is ``None``.
        """
        seed = env['data_seed'] if seed is None else seed
        return split_dataset(dataset, length=length, percent=percent,
                             shuffle=shuffle, seed=seed)

    @staticmethod
    def get_class_subset(dataset: torch.utils.data.Dataset,
                         class_list: int | list[int]) -> torch.utils.data.Subset:
        r"""Get a subset from dataset with certain classes.

        Args:
            dataset (torch.utils.data.Dataset): The entire dataset.
            class_list (int | list[int]): The class list to pick.

        Returns:
            torch.utils.data.Subset:
                The subset with labels in :attr:`class_list`.

        :Example:
            >>> from trojanzoo.utils.data import TensorListDataset
            >>> from trojanzoo.utils.data import get_class_subset
            >>> import torch
            >>>
            >>> data = torch.ones(11, 3, 32, 32)
            >>> targets = list(range(11))
            >>> dataset = TensorListDataset(data, targets)
            >>> subset = get_class_subset(dataset, class_list=[2, 3])
            >>> len(subset)
            2

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.data.get_class_subset`.
        """
        return get_class_subset(dataset=dataset, class_list=class_list)

    def get_dataloader(self, mode: str = None,
                       dataset: torch.utils.data.Dataset = None,
                       batch_size: int = None, shuffle: bool = None,
                       num_workers: int = None, pin_memory: bool = True,
                       drop_last: bool = False, collate_fn: Callable = None,
                       **kwargs) -> torch.utils.data.DataLoader:
        r"""Get dataloader. Call :meth:`get_dataset` if :attr:`dataset` is not provided.

        Args:
            mode (str): Dataset mode (e.g., ``'train'`` or ``'valid'``).
            dataset (torch.utils.data.Dataset): The pytorch dataset.
            batch_size (int):
                Defaults to :attr:`self.batch_size` for ``'train'`` mode
                and :attr:`self.valid_batch_size` for ``'valid'`` mode.
            shuffle (bool): Whether to shuffle.
                Defaults to ``True`` for ``'train'`` mode
                and ``False`` for ``'valid'`` mode.
            num_workers (int): Number of workers for dataloader.
                Defaults to :attr:`self.num_workers`.
            pin_memory (bool): Whether to use pin memory.
                Defaults to ``True`` if there is any GPU available.
            drop_last (bool): Whether drop the last batch if not full size.
                Defaults to ``False``.
            collate_fn (~collections.abc.Callable):
                Passed to :any:`torch.utils.data.DataLoader`.
            **kwargs: Keyword arguments passed to :meth:`get_dataset`
                if :attr:`dataset` is not provided.

        Returns:
            torch.utils.data.DataLoader: The pytorch dataloader.
        """
        if batch_size is None:
            match mode:
                case 'train':
                    batch_size = self.batch_size
                case 'valid':
                    batch_size = self.valid_batch_size
                case _:
                    raise ValueError(f'{mode=}')
        if shuffle is None:
            shuffle = (mode == 'train')
        if num_workers is None:
            num_workers = self.num_workers
        if dataset is None:
            dataset = self.get_dataset(mode=mode, **kwargs)
        pin_memory = pin_memory and env['num_gpus']
        collate_fn = collate_fn or self.collate_fn
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, collate_fn=collate_fn)

    def get_loss_weights(self, file_path: str = None,
                         verbose: bool = True) -> torch.Tensor:
        r"""Calculate :attr:`loss_weights` as reciprocal of data size of each class
        (to mitigate data imbalance).

        Args:
            file_path (str):
                | The file path of saved weights file.
                | If exist, just load the file and return;
                | else, calculate the weights, save and return.
                | Defaults to ``{folder_path}/loss_weights.npy``
            verbose (bool): Whether to print verbose information.
                Defaults to ``True``.

        Returns:
            torch.Tensor: The tensor of loss weights w.r.t. each class.
        """
        file_path = file_path if file_path is not None \
            else os.path.join(self.folder_path, 'loss_weights.npy')
        if os.path.exists(file_path):
            loss_weights = np.load(file_path)
        else:
            if verbose:
                print('Calculating Loss Weights')
            dataset = self.get_dataset('train', transform=None)
            targets = np.array(list(zip(*dataset))[1])
            loss_weights = np.reciprocal(np.bincount(targets).astype(float))
            assert len(loss_weights) == self.num_classes
            np.save(file_path, loss_weights)
            if verbose:
                print('Loss Weights Saved at ', file_path)
        return torch.from_numpy(loss_weights).to(device=env['device'], dtype=torch.float)


def add_argument(parser: argparse.ArgumentParser, dataset_name: str = None,
                 dataset: str | Dataset = None,
                 config: Config = config,
                 class_dict: dict[str, type[Dataset]] = {}
                 ) -> argparse._ArgumentGroup:
    r"""
    | Add dataset arguments to argument parser.
    | For specific arguments implementation, see :meth:`Dataset.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        dataset_name (str): The dataset name.
        dataset (str | Dataset): Dataset instance or dataset name
            (as the alias of `dataset_name`).
        config (Config): The default parameter config,
            which contains the default dataset name if not provided.
        class_dict (dict[str, type[Dataset]]):
            Map from dataset name to dataset class.
            Defaults to ``{}``.
    """
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None \
        else config.full_config['dataset']['default_dataset']
    group = parser.add_argument_group(
        '{yellow}dataset{reset}'.format(**ansi), description=dataset_name)
    try:
        DatasetType = class_dict[dataset_name]
    except KeyError:
        print(f'{dataset_name} not in \n{list(class_dict.keys())}')
        raise
    return DatasetType.add_argument(group)


def create(dataset_name: str = None, dataset: str = None,
           config: Config = config,
           class_dict: dict[str, type[Dataset]] = {},
           **kwargs) -> Dataset:
    r"""
    | Create a dataset instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{data_dir}/{data_type}/{name}'``.
    | For dataset implementation, see :class:`Dataset`.

    Args:
        dataset_name (str): The dataset name.
        dataset (str): The alias of `dataset_name`.
        config (Config): The default parameter config.
        class_dict (dict[str, type[Dataset]]):
            Map from dataset name to dataset class.
            Defaults to ``{}``.
        **kwargs: Keyword arguments
            passed to dataset init method.

    Returns:
        Dataset: Dataset instance.
    """
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None \
        else config.full_config['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)[
        'dataset'].update(kwargs)
    try:
        DatasetType = class_dict[dataset_name]
    except KeyError:
        print(f'{dataset_name} not in \n{list(class_dict.keys())}')
        raise
    if 'folder_path' not in result.keys():
        result['folder_path'] = os.path.join(result['data_dir'],
                                             DatasetType.data_type,
                                             DatasetType.name)
    return DatasetType(**result)
