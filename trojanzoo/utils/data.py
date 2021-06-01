#!/usr/bin/env python3

from .tensor import to_list

import torch
import numpy as np

from typing import TYPE_CHECKING
from typing import Union    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.utils.data


class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor = None, targets: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.targets = to_list(targets)
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index: Union[int, slice]) -> tuple[torch.Tensor, int]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        if indices is None:
            indices = list(range(len(self.dataset)))
        self.indices = indices

    def __getitem__(self, index: Union[int, slice]) -> tuple:
        return (*self.dataset[index], self.indices[index])

    def __len__(self):
        return len(self.dataset)


def dataset_to_list(dataset: torch.utils.data.Dataset, label_only: bool = False,
                    force: bool = True, shuffle: bool = False) -> tuple[list, list[int]]:
    if not force:
        if label_only and 'targets' in dataset.__dict__.keys():
            return None, list(dataset.targets)
        if 'data' in dataset.__dict__.keys() and 'targets' in dataset.__dict__.keys():
            data = dataset.data
            if isinstance(data, np.ndarray):
                data = torch.as_tensor(data)
            if isinstance(data, torch.Tensor):
                if data.max() > 2:
                    data = data.to(dtype=torch.float) / 255
                data = [img for img in data]
            return data, list(dataset.targets)
    data, targets = list(zip(*dataset))[:2]
    if label_only:
        data = None
    else:
        data = list(data)
    targets = list(targets)
    return data, targets


def shuffle_idx(len: int, seed: int = None) -> np.ndarray:
    idx_arr: np.ndarray = np.arange(len)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx_arr)


def sample_batch(dataset: torch.utils.data.Dataset, batch_size: int = None,
                 idx: list[int] = None) -> tuple[list, list[int]]:
    if idx is None:
        assert len(dataset) >= batch_size
        idx = torch.randperm(len(dataset))[:batch_size]
    else:
        assert len(dataset) > max(idx)
    subset = torch.utils.data.Subset(dataset, idx)
    return dataset_to_list(subset)


def split_dataset(dataset: Union[torch.utils.data.Dataset, torch.utils.data.Subset],
                  length: int = None, percent=None, shuffle: bool = True, seed: int = None
                  ) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    assert (length is None) != (percent is None)  # XOR check
    length = length if length is not None else int(len(dataset) * percent)
    indices = np.arange(len(dataset))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    if isinstance(dataset, torch.utils.data.Subset):
        idx = np.array(dataset.indices)
        indices = idx[indices]
        dataset = dataset.dataset
    subset1 = torch.utils.data.Subset(dataset, indices[:length])
    subset2 = torch.utils.data.Subset(dataset, indices[length:])
    return subset1, subset2


def get_class_subset(dataset: torch.utils.data.Dataset, class_list: Union[int, list[int]]) -> torch.utils.data.Subset:
    class_list = [class_list] if isinstance(class_list, int) else class_list
    indices = np.arange(len(dataset))
    if isinstance(dataset, torch.utils.data.Subset):
        idx = np.array(dataset.indices)
        indices = idx[indices]
        dataset = dataset.dataset
    _, targets = dataset_to_list(dataset=dataset, label_only=True)
    idx_bool = np.isin(targets, class_list)
    idx = np.arange(len(dataset))[idx_bool]
    idx = np.intersect1d(idx, indices)
    return torch.utils.data.Subset(dataset, idx)
