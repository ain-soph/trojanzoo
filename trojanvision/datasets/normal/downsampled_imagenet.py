#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet
import trojanvision.utils.datasets.downsampled_imagenet as di
import argparse


class DownsampledImageNet(ImageSet):
    name = 'downsampled_imagenet'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--num_classes', type=int, help='number of classes')
        return group

    def __init__(self, norm_par: dict[str, list[float]] = {
            'mean': [122.68 / 255, 116.66 / 255, 104.01 / 255],
            'std': [63.22 / 255, 61.26 / 255, 65.09 / 255], },
            num_classes: int = 1000, **kwargs):
        self.num_classes = num_classes
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        raise NotImplementedError(
            '\n\n'
            'You need to visit https://image-net.org/download-images.php '
            'to download downsampled image data (raw format).\n'
            'There are direct links to files, but not legal to distribute. '
            'Please apply for access permission and find links yourself.\n\n'
            f'folder_path: {self.folder_path}\n'
            'expected files:\n'
            '{folder_path}/train_data_batch_*\n'
            '{folder_path}/val_data')


class ImageNet16(DownsampledImageNet):
    r"""ImageNet16 dataset introduced by Patryk Chrabaszcz in 2017.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * paper: `A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets`_
        * website: https://patrykchrabaszcz.github.io/Imagenet32/

    Attributes:
        name (str): ``'imagenet16'``
        num_classes (int): Flexible (passed by command line argument, no larger than 1000).
        data_shape (list[int]): ``[3, 16, 16]``

    .. _A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets:
        https://arxiv.org/abs/1707.08819
    """
    name = 'imagenet16'
    data_shape = [3, 16, 16]

    def _get_org_dataset(self, mode: str, **kwargs):
        assert mode in ['train', 'valid']
        return di.ImageNet16(root=self.folder_path, train=(mode == 'train'),
                             num_classes=self.num_classes if self.num_classes < 1000 else None, **kwargs)


class ImageNet32(DownsampledImageNet):
    r"""ImageNet32 dataset introduced by Patryk Chrabaszcz in 2017.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * paper: `A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets`_
        * website: https://patrykchrabaszcz.github.io/Imagenet32/

    Attributes:
        name (str): ``'imagenet32'``
        num_classes (int): Flexible (passed by command line argument, no larger than 1000).
        data_shape (list[int]): ``[3, 32, 32]``

    .. _A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets:
        https://arxiv.org/abs/1707.08819
    """
    name = 'imagenet32'
    data_shape = [3, 32, 32]

    def _get_org_dataset(self, mode: str, **kwargs):
        assert mode in ['train', 'valid']
        return di.ImageNet32(root=self.folder_path, train=(mode == 'train'),
                             num_classes=self.num_classes if self.num_classes < 1000 else None, **kwargs)
