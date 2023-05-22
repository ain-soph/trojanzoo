#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import numpy as np
import os

from collections.abc import Callable

from trojanvision import __file__ as root_file
root_dir = os.path.dirname(root_file)


class STL10(ImageSet):
    r"""STL10 dataset.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.STL10`
        * paper: `An Analysis of Single-Layer Networks in Unsupervised Feature Learning`_
        * website: https://cs.stanford.edu/~acoates/stl10/

    Attributes:
        name (str): ``'stl10'``
        num_classes (int): ``10``
        data_shape (list[int]): ``[3, 256, 256]``
        norm_par (dict[str, list[float]]):
            ``{'mean': [0.507, 0.487, 0.441], 'std': [0.267, 0.256, 0.276]}``

    .. _An Analysis of Single-Layer Networks in Unsupervised Feature Learning:
        https://cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """

    name: str = 'stl10'
    num_classes: int = 10
    data_shape = [3, 256, 256]

    def __init__(self, norm_par: dict[str, list[int]] = {'mean': [0.507, 0.487, 0.441],
                                                         'std': [0.267, 0.256, 0.276], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.STL10(root=self.folder_path, split='train', download=True)
        datasets.STL10(root=self.folder_path, split='test', download=True)
        datasets.STL10(root=self.folder_path, split='unlabeled', download=True)

    def _get_org_dataset(self, mode, **kwargs):
        if mode == 'valid':
            mode = 'test'
        return STL10Dataset(root=self.folder_path, split=mode, **kwargs)

    def get_transform(self, mode: str, normalize: bool = None
                      ) -> transforms.Compose:
        r"""Get dataset transform.

        Args:
            mode (str): The dataset mode (e.g., ``'train' | 'valid' | 'unlabeled' | 'train+unlabeled'``).
            normalize (bool | None):
                Whether to use :any:`torchvision.transforms.Normalize`
                in dataset transform. Defaults to ``self.normalize``.

        Returns:
            torchvision.transforms.Compose: The transform sequence.
        """
        normalize = normalize if normalize is not None else self.normalize

        match mode:
            case 'train' | 'unlabeled' | 'train+unlabeled':
                transform = transforms.Compose([
                    transforms.Resize((272, 272)),
                    transforms.RandomRotation(15,),
                    transforms.RandomCrop(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float)])
            case 'valid':
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float)])
            case _:
                raise NotImplementedError(mode)
        if normalize and self.norm_par is not None:
            transform.transforms.append(transforms.Normalize(
                mean=self.norm_par['mean'], std=self.norm_par['std']))
        return transform


class STL10Dataset(datasets.STL10):
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
        ["unlabeled_y.bin", None],
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: int | None = None,
        transform: Callable | None = None,
        target_transform:  Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split=split, folds=folds,
                         transform=transform, target_transform=target_transform,
                         download=download)
        match self.split:
            case "train+unlabeled":
                labels = self.read_labels()
                self.labels = np.concatenate((self.labels[:-labels.shape[0]], labels))
            case "unlabeled":
                self.labels = self.read_labels()

    @staticmethod
    def read_labels() -> np.ndarray:
        path_to_labels = os.path.normpath(os.path.join(root_dir, 'data', 'stl10', 'unlabeled_y.bin'))
        with open(path_to_labels, "rb") as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based
        return labels
