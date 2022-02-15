#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets


class MNIST(ImageSet):
    r"""MNIST dataset. It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        :any:`torchvision.datasets.MNIST`

    Attributes:
        name (str): ``'mnist'``
        num_classes (int): ``10``
        data_shape (list[int]): ``[1, 28, 28]``
        norm_par (dict[str, list[float]]):
            ``{'mean': [0.1307], 'std': [0.3081]}``
    """

    name: str = 'mnist'
    num_classes: int = 10
    data_shape = [1, 28, 28]

    def __init__(self, norm_par: dict[str, list[int]] = {'mean': [0.1307], 'std': [0.3081]}, **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.MNIST(root=self.folder_path, train=True, download=True)
        datasets.MNIST(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode, **kwargs):
        assert mode in ['train', 'valid']
        return datasets.MNIST(root=self.folder_path, train=(mode == 'train'), **kwargs)
