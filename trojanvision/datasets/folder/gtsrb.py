#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
import torch
import torchvision.transforms as transforms


class GTSRB(ImageFolder):
    r"""GTSRB dataset introduced by Johannes Stallkamp in 2011.
    It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * paper: `The German Traffic Sign Recognition Benchmark\: A multi-class classification competition`_
        * website: https://benchmark.ini.rub.de/gtsrb_dataset.html

    Attributes:
        name (str): ``'gtsrb'``
        num_classes (int): ``43``
        data_shape (list[int]): ``[3, 32, 32]``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.3403, 0.3121, 0.3214],``
            | ``'std'  : [0.2724, 0.2608, 0.2669]}``
        valid_set (bool): ``False``
        loss_weights (bool): ``True``

    .. _The German Traffic Sign Recognition Benchmark\: A multi-class classification competition:
        https://ieeexplore.ieee.org/document/6033395
    """
    name = 'gtsrb'
    data_shape = [3, 32, 32]
    num_classes = 43
    valid_set = False
    url = {'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
    md5 = {'train': '513f3c79a4c5141765e10e952eaa2478'}
    org_folder_name = {'train': 'GTSRB/Training'}

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.3403, 0.3121, 0.3214],
                                                           'std': [0.2724, 0.2608, 0.2669], },
                 loss_weights: bool = True, **kwargs):
        return super().__init__(norm_par=norm_par, loss_weights=loss_weights, **kwargs)

    def get_transform(self, mode: str) -> transforms.Compose:
        if mode != 'train':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float)])
        else:
            transform = super().get_transform(mode=mode)
        return transform
