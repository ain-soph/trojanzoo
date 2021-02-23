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
                    force: bool = False) -> tuple[list, list[int]]:
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
    data, targets = zip(*dataset)[:2]
    if label_only:
        data = None
    else:
        data = list(data)
    targets = list(targets)
    return data, targets


def sample_batch(dataset: torch.utils.data.Dataset, batch_size: int = None,
                 idx: list[int] = None) -> tuple[list, list[int]]:
    if idx is None:
        assert len(dataset) >= batch_size
        idx = torch.randperm(len(dataset))[:batch_size]
    else:
        assert len(dataset) > max(idx)
    subset = torch.utils.data.Subset(dataset, idx)
    return dataset_to_list(subset)
