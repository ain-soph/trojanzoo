#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets


class CIFAR10(ImageSet):
    r"""CIFAR10 dataset introduced by Alex Krizhevsky in 2009.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.CIFAR10`
        * paper: `Learning Multiple Layers of Features from Tiny Images`_
        * website: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes:
        name (str): ``'cifar10'``
        num_classes (int): ``10``
        data_shape (list[int]): ``[3, 32, 32]``
        class_names (list[str]):
            | ``['airplane', 'automobile', 'bird', 'cat', 'deer',``
            | ``'dog', 'frog', 'horse', 'ship', 'truck']``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.49139968, 0.48215827, 0.44653124],``
            | ``'std'  : [0.24703233, 0.24348505, 0.26158768]}``

    .. _Learning Multiple Layers of Features from Tiny Images:
        https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.49139968, 0.48215827, 0.44653124],
                                                           'std': [0.24703233, 0.24348505, 0.26158768], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.CIFAR10(root=self.folder_path, train=True, download=True)
        datasets.CIFAR10(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs) -> datasets.CIFAR10:
        assert mode in ['train', 'valid']
        return datasets.CIFAR10(root=self.folder_path, train=(mode == 'train'), **kwargs)


class CIFAR100(CIFAR10):
    r"""CIFAR100 dataset. It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.CIFAR100`
        * paper: `Learning Multiple Layers of Features from Tiny Images`_
        * website: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes:
        name (str): ``'cifar100'``
        num_classes (int): ``100``
        data_shape (list[int]): ``[3, 32, 32]``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.49139968, 0.48215827, 0.44653124],``
            | ``'std'  : [0.24703233, 0.24348505, 0.26158768]}``

    .. _Learning Multiple Layers of Features from Tiny Images:
        https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    name = 'cifar100'
    num_classes = 100

    def initialize(self):
        datasets.CIFAR100(root=self.folder_path, train=True, download=True)
        datasets.CIFAR100(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs) -> datasets.CIFAR100:
        assert mode in ['train', 'valid']
        return datasets.CIFAR100(root=self.folder_path, train=(mode == 'train'), **kwargs)
