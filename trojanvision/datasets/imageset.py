#!/usr/bin/env python3

from trojanzoo.datasets import Dataset
from trojanvision.environ import env
from trojanvision.utils.transform import (get_transform_bit,
                                          get_transform_imagenet,
                                          get_transform_cifar,
                                          RandomMixup,
                                          RandomCutmix)

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
import argparse
import os

from typing import TYPE_CHECKING
from typing import Iterable
from torchvision.datasets import VisionDataset  # TODO: python 3.10
import PIL.Image as Image
if TYPE_CHECKING:
    import torch.utils.data


class ImageSet(Dataset):
    r"""
    | The basic class representing an image dataset.
    | It inherits :class:`trojanzoo.datasets.Dataset`.

    Note:
        This is the implementation of dataset.
        For users, please use :func:`create` instead, which is more user-friendly.

    Args:
        norm_par (dict[str, list[float]]):
            Data normalization parameters of ``'mean'`` and ``'std'``
            (e.g., ``{'mean': [0.5, 0.4, 0.6], 'std': [0.2, 0.3, 0.1]}``).
            Defaults to ``None``.
        normalize (bool): Whether to use :any:`torchvision.transforms.Normalize`
            in dataset transform. Otherwise, use it as model preprocess layer.
        transform (str): The dataset transform type.

            * ``None |'none'`` (:any:`torchvision.transforms.PILToTensor`
              and :any:`torchvision.transforms.ConvertImageDtype`)
            * ``'bit'`` (transform used in BiT network)
            * ``'pytorch'`` (pytorch transform used in ImageNet training).

            Defaults to ``None``.

            Note:
                See :meth:`get_transform()` to get more details.
        auto_augment (bool): Whether to use
            :any:`torchvision.transforms.AutoAugment`.
            Defaults to ``False``.
        mixup (bool): Whether to use
            :class:`trojanvision.utils.transforms.RandomMixup`.
            Defaults to ``False``.
        mixup_alpha (float): :attr:`alpha` passed to
            :class:`trojanvision.utils.transforms.RandomMixup`.
            Defaults to ``0.0``.
        cutmix (bool): Whether to use
            :class:`trojanvision.utils.transforms.RandomCutmix`.
            Defaults to ``False``.
        cutmix_alpha (float): :attr:`alpha` passed to
            :class:`trojanvision.utils.transforms.RandomCutmix`.
            Defaults to ``0.0``.
        cutout (bool): Whether to use
            :class:`trojanvision.utils.transforms.Cutout`.
            Defaults to ``False``.
        cutout_length (int): Cutout length. Defaults to ``None``.
        **kwargs: keyword argument passed to
            :class:`trojanzoo.datasets.Dataset`.

    Attributes:
        data_type (str): Defaults to ``'image'``.
        num_classes (int): Defaults to ``1000``.
        data_shape (list[int]): The shape of image data ``[C, H, W]``.
            Defaults to ``[3, 224, 224]``.
    """

    name: str = 'imageset'
    data_type: str = 'image'
    num_classes = 1000
    data_shape = [3, 224, 224]

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        r"""Add image dataset arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete dataset class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.

        See Also:
            :meth:`trojanzoo.datasets.Dataset.add_argument()`
        """
        super().add_argument(group)
        group.add_argument(
            '--dataset_normalize', dest='normalize', action='store_true',
            help='use transforms.Normalize in dataset transform. '
            '(It\'s used in model as the first layer by default.)')
        group.add_argument('--transform', choices=[None, 'none', 'bit', 'pytorch'])
        group.add_argument('--auto_augment', action='store_true',
                           help='use auto augment')
        group.add_argument('--mixup', action='store_true', help='use mixup')
        group.add_argument('--mixup_alpha', type=float, help='mixup alpha (default: 0.0)')
        group.add_argument('--cutmix', action='store_true', help='use cutmix')
        group.add_argument('--cutmix_alpha', type=float, help='cutmix alpha (default: 0.0)')
        group.add_argument('--cutout', action='store_true', help='use cutout')
        group.add_argument('--cutout_length', type=int, help='cutout length')
        return group

    def __init__(self, norm_par: dict[str, list[float]] = None,
                 normalize: bool = False, transform: str = None,
                 auto_augment: bool = False,
                 mixup: bool = False, mixup_alpha: float = 0.0,
                 cutmix: bool = False, cutmix_alpha: float = 0.0,
                 cutout: bool = False, cutout_length: int = None,
                 **kwargs):
        self.norm_par: dict[str, list[float]] = norm_par
        self.normalize = normalize
        self.transform = transform
        self.auto_augment = auto_augment
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix = cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutout = cutout
        self.cutout_length = cutout_length

        mixup_transforms = []
        if mixup:
            mixup_transforms.append(RandomMixup(self.num_classes, p=1.0, alpha=mixup_alpha))
        if cutmix:
            mixup_transforms.append(RandomCutmix(self.num_classes, p=1.0, alpha=cutmix_alpha))
        if len(mixup_transforms):
            mixupcutmix = mixup_transforms[0] if len(mixup_transforms) == 1 \
                else transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
                return mixupcutmix(*default_collate(batch))  # noqa: E731
            self.collate_fn = collate_fn

        super().__init__(**kwargs)
        self.param_list['imageset'] = ['data_shape', 'norm_par',
                                       'normalize', 'transform',
                                       'auto_augment']
        if cutout:
            self.param_list['imageset'].append('cutout_length')

        if mixup:
            self.param_list['imageset'].append('mixup_alpha')
        if cutmix:
            self.param_list['imageset'].append('cutmix_alpha')

    def get_transform(self, mode: str, normalize: bool = None
                      ) -> transforms.Compose:
        r"""Get dataset transform based on :attr:`self.transform`.

            * ``None |'none'`` (:any:`torchvision.transforms.PILToTensor`
              and :any:`torchvision.transforms.ConvertImageDtype`)
            * ``'bit'`` (transform used in BiT network)
            * ``'pytorch'`` (pytorch transform used in ImageNet training).

        Args:
            mode (str): The dataset mode (e.g., ``'train' | 'valid'``).
            normalize (bool | None):
                Whether to use :any:`torchvision.transforms.Normalize`
                in dataset transform. Defaults to ``self.normalize``.

        Returns:
            torchvision.transforms.Compose: The transform sequence.
        """
        normalize = normalize if normalize is not None else self.normalize
        if self.transform == 'bit':
            return get_transform_bit(mode, self.data_shape)
        elif self.data_shape == [3, 224, 224]:
            transform = get_transform_imagenet(
                mode, use_tuple=self.transform != 'pytorch',
                auto_augment=self.auto_augment)
        elif self.transform != 'none' and self.data_shape in ([3, 16, 16], [3, 32, 32]):
            transform = get_transform_cifar(
                mode, auto_augment=self.auto_augment,
                cutout=self.cutout, cutout_length=self.cutout_length,
                data_shape=self.data_shape)
        else:
            transform = transforms.Compose([transforms.PILToTensor(),
                                            transforms.ConvertImageDtype(torch.float)])
        if normalize and self.norm_par is not None:
            transform.transforms.append(transforms.Normalize(
                mean=self.norm_par['mean'], std=self.norm_par['std']))
        return transform

    @staticmethod
    def get_data(data: tuple[torch.Tensor, torch.Tensor],
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Process image data.
        Defaults to put input and label on ``env['device']`` with ``non_blocking``
        and transform label to ``torch.LongTensor``.

        Args:
            data (tuple[torch.Tensor, torch.Tensor]): Tuple of batched input and label.
            **kwargs: Any keyword argument (unused).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]):
                Tuple of batched input and label on ``env['device']``.
                Label is transformed to ``torch.LongTensor``.
        """
        return (data[0].to(env['device'], non_blocking=True),
                data[1].to(env['device'], dtype=torch.long, non_blocking=True))

    def make_folder(self, img_type: str = '.png', **kwargs):
        r"""Save the dataset to ``self.folder_path``
        as :class:`trojanvision.datasets.ImageFolder` format.

        ``'{self.folder_path}/{self.name}/{mode}/{class_name}/{img_idx}.png'``

        Args:
            img_type (str): The image types to save. Defaults to ``'.png'``.
        """
        mode_list: list[str] = [
            'train', 'valid'] if self.valid_set else ['train']
        class_names = getattr(self, 'class_names',
                              [str(i) for i in range(self.num_classes)])
        for mode in mode_list:
            dataset: VisionDataset = self.get_org_dataset(mode, transform=None)
            class_counters = [0] * self.num_classes
            for image, target_class in list(dataset):
                image: Image.Image
                target_class: int
                class_name = class_names[target_class]
                _dir = os.path.join(
                    self.folder_path, self.name, mode, class_name)
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                image.save(os.path.join(
                    _dir, f'{class_counters[target_class]}{img_type}'))
                class_counters[target_class] += 1
