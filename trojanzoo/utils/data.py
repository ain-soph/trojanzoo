#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, Subset
import numpy as np

from typing import TYPE_CHECKING, overload
from typing import Optional, Union    # TODO: python 3.10
if TYPE_CHECKING:
    pass


class TensorListDataset(Dataset):
    r"""The dataset class that has a :any:`torch.Tensor` as inputs
    and :any:`list`\[:any:`int`\] as labels.
    It inherits :any:`torch.utils.data.Dataset`.

    Args:
        data (torch.Tensor): The inputs.
        targets (list[int]): The labels.
        **kwargs: Keyword arguments passed to
            :any:`torch.utils.data.Dataset`.

    :Example:
        >>> from trojanzoo.utils.data import TensorListDataset
        >>> import torch
        >>> data = torch.ones(10, 3, 32, 32)
        >>> targets = list(range(10))
        >>> dataset = TensorListDataset(data, targets)
        >>> x, y = dataset[3]
        >>> x.shape
        torch.Size([3, 32, 32])
        >>> y
        3
    """

    def __init__(self, data: torch.Tensor = None,
                 targets: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.targets = targets
        assert len(self.data) == len(self.targets)
        self.__length = len(self.targets)

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[torch.Tensor, list[int]]:
        ...

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return self.__length


def dataset_to_list(dataset: Dataset, label_only: bool = False,
                    force: bool = True) -> tuple[Optional[list], list[int]]:
    r"""transform a :any:`torch.utils.data.Dataset` to ``(data, targets)`` lists.

    Args:
        dataset (torch.utils.data.Dataset): The dataset.
        label_only (bool): Whether to only return the ``targets``.
            If ``True``, the first return element ``data`` will be ``None``.
            Defaults to ``False``.
        force (bool): Whether to force traversing the dataset
            to get data and targets.
            If ``False``, it will return
            ``(datasets.data, datasets.targets)`` if possible.
            It should be ``True`` when :attr:`dataset` has transform.
            Defaults to ``True``.

    Returns:
        (Optional[list], list[int]): The tuple of ``(data, targets)``.

    :Example:
        >>> from trojanzoo.utils.data import dataset_to_list
        >>> from torchvision.datasets import MNIST
        >>> from torchvision.transforms import ToTensor
        >>> dataset = MNIST('./', train=False, download=True)
        >>> data, targets = dataset_to_list(dataset)
        >>> type(data[0])
        <PIL.Image.Image image mode=L size=28x28 at 0x19FCF226D30>
        >>> data, targets = dataset_to_list(dataset, force=False)
        >>> type(data[0])
        <class 'torch.Tensor'>
    """  # noqa: E501
    if not force:
        targets = list(dataset.targets)
        if label_only and hasattr(dataset, 'targets'):
            return None, targets
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            data = dataset.data
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(data, torch.Tensor):
                if data.max() > 2:
                    data = data.to(dtype=torch.float) / 255
                data = [img for img in data]
            return data, targets
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


def sample_batch(dataset: Dataset, batch_size: int = None,
                 idx: list[int] = None) -> tuple[list, list[int]]:
    if idx is None:
        assert len(dataset) >= batch_size
        idx = torch.randperm(len(dataset))[:batch_size]
    else:
        assert len(dataset) > max(idx)
    subset = Subset(dataset, idx)
    return dataset_to_list(subset)


def split_dataset(dataset: Union[Dataset, Subset],
                  length: int = None, percent=None, shuffle: bool = True, seed: int = None
                  ) -> tuple[Subset, Subset]:
    assert (length is None) != (percent is None)  # XOR check
    length = length if length is not None else int(len(dataset) * percent)
    indices = np.arange(len(dataset))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    if isinstance(dataset, Subset):
        idx = np.array(dataset.indices)
        indices = idx[indices]
        dataset = dataset.dataset
    subset1 = Subset(dataset, indices[:length])
    subset2 = Subset(dataset, indices[length:])
    return subset1, subset2


def get_class_subset(dataset: Dataset, class_list: Union[int, list[int]]) -> Subset:
    class_list = [class_list] if isinstance(class_list, int) else class_list
    indices = np.arange(len(dataset))
    if isinstance(dataset, Subset):
        idx = np.array(dataset.indices)
        indices = idx[indices]
        dataset = dataset.dataset
    _, targets = dataset_to_list(dataset=dataset, label_only=True)
    idx_bool = np.isin(targets, class_list)
    idx = np.arange(len(dataset))[idx_bool]
    idx = np.intersect1d(idx, indices)
    return Subset(dataset, idx)
