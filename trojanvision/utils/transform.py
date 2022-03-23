#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.nn.functional import one_hot

import math


__all__ = ['RandomMixup', 'RandomCutmix', 'Cutout',
           'get_transform_bit',
           'get_transform_imagenet',
           'get_transform_cifar']


class RandomMixup(nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """  # noqa: E501

    def __init__(self, num_classes: int, p: float = 0.5,
                 alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, ("Please provide a valid positive value"
                                 " for the num_classes.")
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: torch.Tensor,
                target: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (torch.Tensor): Float tensor of size (B, C, H, W)
            target (torch.Tensor): Integer tensor of size (B, )
        Returns:
            torch.Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(
                f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(
                f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = one_hot(target, num_classes=self.num_classes)
            target = target.to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one
        # instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)


class RandomCutmix(nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """  # noqa: E501

    def __init__(self, num_classes: int, p: float = 0.5,
                 alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, ("Please provide a valid positive value"
                                 " for the num_classes.")
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: torch.Tensor,
                target: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (torch.Tensor): Float tensor of size (B, C, H, W)
            target (torch.Tensor): Integer tensor of size (B, )
        Returns:
            torch.Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(
                f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(
                f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = one_hot(target, num_classes=self.num_classes)
            target = target.to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one
        # instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12
        # (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[..., y1:y2, x1:x2] = batch_rolled[..., y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)


def cutout(img: torch.Tensor, length: int | tuple[int, int] | torch.Tensor,
           fill_values: float | torch.Tensor = 0.0) -> torch.Tensor:
    if isinstance(length, int):
        length = (length, length)
    h, w = img.size(-2), img.size(-1)
    mask = torch.ones(h, w, dtype=torch.bool, device=img.device)
    device = length.device if isinstance(length, torch.Tensor) else img.device
    y = torch.randint(0, h, [1], device=device)
    x = torch.randint(0, w, [1], device=device)
    first_half = [length[0] // 2, length[1] // 2]
    second_half = [length[0] - first_half[0], length[1] - first_half[1]]

    y1 = max(y - first_half[0], 0)
    y2 = min(y + second_half[0], h)
    x1 = max(x - first_half[1], 0)
    x2 = min(x + second_half[1], w)
    mask[y1: y2, x1: x2] = False
    return mask * img + ~mask * fill_values


class Cutout(nn.Module):
    def __init__(self, length: int,
                 fill_values: float | torch.Tensor = 0.0):
        super().__init__()
        self.length = length
        self.fill_values = fill_values

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return cutout(img, self.length, self.fill_values)


def get_transform_bit(mode: str, data_shape: list[int]) -> transforms.Compose:
    hyperrule = data_shape[-2] * data_shape[-1] < 96 * 96
    precrop, crop = (160, 128) if hyperrule else (512, 480)
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((precrop, precrop)),
            transforms.RandomCrop((crop, crop)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)])
    return transform


def get_transform_imagenet(mode: str, use_tuple: bool = False,
                           auto_augment: bool = False) -> transforms.Compose:
    if mode == 'train':
        transform_list = [
            transforms.RandomResizedCrop((224, 224) if use_tuple else 224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # noqa
        ]
        if auto_augment:
            transform_list.append(transforms.AutoAugment(
                transforms.AutoAugmentPolicy.IMAGENET))
        transform_list.append(transforms.PILToTensor())
        transform_list.append(transforms.ConvertImageDtype(torch.float))
        transform = transforms.Compose(transform_list)
    else:
        # TODO: torchvision.prototypes.transforms._presets.ImageClassificationEval
        transform = transforms.Compose([
            transforms.Resize((256, 256) if use_tuple else 256),
            transforms.CenterCrop((224, 224) if use_tuple else 224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)])
    return transform


def get_transform_cifar(mode: str, auto_augment: bool = False,
                        cutout: bool = False, cutout_length: int = None,
                        data_shape: list[int] = [3, 32, 32]
                        ) -> transforms.Compose:
    if mode != 'train':
        return transforms.Compose([transforms.PILToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
    cutout_length = cutout_length or data_shape[-1] // 2
    transform_list = [
        transforms.RandomCrop(data_shape[-2:], padding=data_shape[-1] // 8),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_augment:
        transform_list.append(transforms.AutoAugment(
            transforms.AutoAugmentPolicy.CIFAR10))
    transform_list.append(transforms.PILToTensor())
    transform_list.append(transforms.ConvertImageDtype(torch.float))
    if cutout:
        transform_list.append(Cutout(cutout_length))
    return transforms.Compose(transform_list)
