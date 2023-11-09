#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets


class CelebA(ImageSet):
    r"""CelebA dataset.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.CelebA`
        * paper: `Deep Learning Face Attributes in the Wild`_
        * website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Attributes:
        name (str): ``'celeba'``
        num_classes (int): ``10177``
        data_shape (list[int]): ``[3, 64, 64]``

    .. _Deep Learning Face Attributes in the Wild:
        https://arxiv.org/abs/1411.7766
    """

    name: str = 'celeba'
    num_classes: int = 10177
    data_shape = [3, 64, 64]

    def __init__(self, norm_par: dict[str, list[int]] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}, **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.CelebA(root=self.folder_path, split='all', download=True)

    def _get_org_dataset(self, mode, **kwargs):
        return datasets.CelebA(root=self.folder_path, split=mode, target_type='identity', **kwargs)
