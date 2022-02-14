#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, Subset
import numpy as np

from typing import overload
from typing import Optional, Union    # TODO: python 3.10


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
            It should be ``True`` when there is extra operations
            for ``__getitem__`` or ``transform``.
            Defaults to ``True``.

    Returns:
        (list | None, list[int]): The tuple of ``(data, targets)``.

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


def sample_batch(dataset: Dataset, idx: list[int] = None,
                 batch_size: int = None) -> tuple[list, list[int]]:
    r"""Sample a batch from dataset by calling

    .. parsed-literal::
        :any:`dataset_to_list`\(:any:`torch.utils.data.Subset`\(dataset, idx))

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample.
        idx (list[int]): The index list of each sample in dataset.
            If ``None``, randomly sample a batch with given :attr:`batch_size`.
            Defaults to ``None``.
        batch_size (int): The batch size to sample
            when :attr:`idx` is ``None``.
            Defaults to ``None``.

    Returns:
        (list, list[int]): The tuple of sampled batch ``(data, targets)``.

    :Example:
        >>> from trojanzoo.utils.data import TensorListDataset, sample_batch
        >>> import torch
        >>> data = torch.ones(10, 3, 32, 32)
        >>> targets = list(range(10))
        >>> dataset = TensorListDataset(data, targets)
        >>> x, y = sample_batch(dataset, [1, 2])
        >>> torch.stack(x).shape
        torch.Size([2, 3, 32, 32])
        >>> y
        [1, 2]
        >>> x, y = sample_batch(dataset, batch_size=4)
        >>> y
        [6, 3, 2, 5]
    """
    if idx is None:
        assert len(dataset) >= batch_size
        idx = torch.randperm(len(dataset))[:batch_size]
    else:
        assert len(dataset) > max(idx)
    subset = Subset(dataset, idx)
    return dataset_to_list(subset)


def split_dataset(dataset: Union[Dataset, Subset],
                  length: int = None, percent: float = None,
                  shuffle: bool = True, seed: int = None
                  ) -> tuple[Subset, Subset]:
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
            Defaults to ``None``.

    Returns:
        (torch.utils.data.Subset, torch.utils.data.Subset):
            The two splitted subsets.

    :Example:
        >>> from trojanzoo.utils.data import TensorListDataset, split_dataset
        >>> import torch
        >>> data = torch.ones(11, 3, 32, 32)
        >>> targets = list(range(11))
        >>> dataset = TensorListDataset(data, targets)
        >>> set1, set2 = split_dataset(dataset, length=3)
        >>> len(set1), len(set2)
        (3, 8)
        >>> set3, set4 = split_dataset(dataset, percent=0.5)
        >>> len(set3), len(set4)
        (5, 6)

    Note:
        This is the implementation of :meth:`trojanzoo.datasets.Dataset.split_dataset`.
        The difference is that this method will NOT set :attr:`seed`
        as ``env['data_seed']`` when it is ``None``.
    """
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


def get_class_subset(dataset: Dataset,
                     class_list: Union[int, list[int]]
                     ) -> Subset:
    r"""Get a subset from dataset with certain classes.

    Args:
        dataset (torch.utils.data.Dataset): The entire dataset.
        class_list (int | list[int]): The class list to pick.

    Returns:
        torch.utils.data.Subset: The subset with labels in :attr:`class_list`.

    :Example:
        >>> from trojanzoo.utils.data import TensorListDataset
        >>> from trojanzoo.utils.data import get_class_subset
        >>> import torch
        >>> data = torch.ones(11, 3, 32, 32)
        >>> targets = list(range(11))
        >>> dataset = TensorListDataset(data, targets)
        >>> subset = get_class_subset(dataset, class_list=[2, 3])
        >>> len(subset)
        2
    """
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
