#!/usr/bin/env python3

from .utils import _Resource

from torchdata.datapipes.iter import Filter, IterDataPipe, Mapper
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, GDriveResource, OnlineResource
from torchvision.prototype.datapoints import Label
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.datasets._builtin.imagenet import _info  # TODO: unstable

import os
import pathlib
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, BinaryIO, Generic, Literal, Sequence, TypeVar


__all__ = ['ImageNetC', 'ImageNetP']


CDistortionCategoryType = Literal['noise', 'blur', 'weather', 'digital', 'extra']
PDistortionCategoryType = Literal['noise', 'blur', 'weather', 'digital', 'validation',
                                  'harder_noise', 'harder_noise_validation']
# DistortionCategoryType = TypeVar('DistortionCategoryType', bound=str)
DistortionCategoryType = TypeVar('DistortionCategoryType', CDistortionCategoryType, PDistortionCategoryType)


class _ImageNetBase(Dataset, Generic[DistortionCategoryType], ABC):
    _DISTORTIONS: dict[DistortionCategoryType, list[str]]
    _RESOURCES: dict[DistortionCategoryType, _Resource]

    @cached_property
    def distortion_map(self) -> dict[str, DistortionCategoryType]:
        return {distort: k for k, v in self._DISTORTIONS.items() for distort in v}

    def __init__(self, root: str | pathlib.Path,
                 distortion_name: str,
                 skip_integrity_check: bool = False) -> None:
        self.distortion_name = distortion_name
        info = _info()
        categories, wnids = info["categories"], info["wnids"]
        self._categories = categories
        self._wnids = wnids
        self._wnid_to_category = dict(zip(wnids, categories))
        super().__init__(root=root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> list[OnlineResource]:
        distortion_category = self.distortion_map[self.distortion_name]
        return [GDriveResource(**self._RESOURCES[distortion_category])]

    def _datapipe(self, resource_dps: list[IterDataPipe]) -> IterDataPipe[dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, self._filter)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    @abstractmethod
    def _filter(self, data: tuple[str, Any]) -> bool:
        ...

    @abstractmethod
    def _prepare_sample(
        self,
        data: tuple[str, BinaryIO],
    ) -> dict[str, Any]:
        ...

    def __len__(self) -> int:
        return 50_000


class ImageNetC(_ImageNetBase[CDistortionCategoryType]):
    _DISTORTIONS = {
        'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'weather': ['snow', 'frost', 'fog', 'brightness'],
        'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
        'extra': ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate'],
    }
    _RESOURCES = {
        'noise': dict(
            id='1w05DJwhGz66zXTA0WK1ie9R54-ZmCtGB',
            file_name='noise.tar',
        ),
        'blur': dict(
            id='15aiZpiQpQzYwWWSSpwKHf7wKo65j-oF4',
            file_name='blur.tar',
        ),
        'weather': dict(
            id='1IGdjgLrQocafIIYLs_r_skfOq24oNbB6',
            file_name='weather.tar',
        ),
        'digital': dict(
            id='15vLMParMqQDpDe34qXTq1eAwZCK4OU_K',
            file_name='digital.tar',
        ),
        'extra': dict(
            id='1LjYf2LMhSPfSdCYR9DFZj2N24ix84fds',
            file_name='extra.tar',
        ),
    }

    def __init__(self, root: str | pathlib.Path, distortion_name: str,
                 severity: int | Sequence[int] = (1, 2, 3, 4, 5),
                 skip_integrity_check: bool = False) -> None:
        self.severity: tuple[int, ...] = (severity,) if isinstance(severity, int) else tuple(severity)
        super().__init__(root, distortion_name, skip_integrity_check=skip_integrity_check)

    def _filter(self, data: tuple[str, Any]) -> bool:
        # blur.tar/defocus_blur/4/n03884397/ILSVRC2012_val_00018337.JPEG
        split_list = os.path.split(data[0])
        distortion_name, severity = split_list[-4], split_list[-3]
        return distortion_name == self.distortion_name and severity in self.severity

    def _prepare_sample(
        self,
        data: tuple[str, BinaryIO],
    ) -> dict[str, Any]:
        path, buffer = data
        wnid = os.path.split(data[0])[-2]
        label = Label.from_category(self._wnid_to_category[wnid], categories=self._categories)
        return dict(label=label, wnid=wnid, path=path,
                    image=EncodedImage.from_file(buffer))


class ImageNetP(_ImageNetBase[PDistortionCategoryType]):
    _DISTORTIONS = {
        'noise': ['gaussian_noise', 'impulse_noise'],
        'blur': ['motion_blur', 'zoom_blur'],
        'weather': ['brightness', 'snow'],
        'digital': ['translate', 'rotate', 'tilt', 'scale'],
        'validation': ['speckle_noise', 'gaussian_blur', 'spatter', 'shear'],
        # 'harder_noise': ['harder_noise'],
        # 'harder_noise_validation': ['harder_noise_validation'],
    }
    _RESOURCES = {
        'noise': dict(
            id='1T8UCwCb1oe68-wqeUHUWnWE3-nVzHiOL',
            file_name='noise.tar',
        ),
        'blur': dict(
            id='1828AvOvFBfzlI7EVdOYDZRGOWh2GEds8',
            file_name='blur.tar',
        ),
        'weather': dict(
            id='1AaklXwxUgXbDSw08YKD8rAeqDchs7omB',
            file_name='weather.tar',
        ),
        'digital': dict(
            id='1lafcnKJ1D_cRpwFYRe0prgOUjlspdrbZ',
            file_name='digital.tar',
        ),
        'validation': dict(
            id='1i4MwC5-7scfop1LOmjVUJcvyO0wjg0L-',
            file_name='validation.tar',
        ),
        'harder_noise': dict(
            id='1AkrFIhCufEq70MN-P8uuQxaMX8wCIUz4',
            file_name='harder_noise.tar',
        ),
        'harder_noise_validation': dict(
            id='1kbLYznkjXIlGkxtniIhPsuqP2fuYfkZT',
            file_name='harder_noise_validation.tar',
        ),
    }

    def _filter(self, data: tuple[str, Any]) -> bool:
        # weather.tar/brightness/n03032252/ILSVRC2012_val_00015013.mp4
        split_list = os.path.split(data[0])
        distortion_name = split_list[-3]
        return distortion_name == self.distortion_name

    def _prepare_sample(
        self,
        data: tuple[str, BinaryIO],
    ) -> dict[str, Any]:
        path, buffer = data
        wnid = os.path.split(data[0])[-2]
        label = Label.from_category(self._wnid_to_category[wnid], categories=self._categories)
        return dict(label=label, wnid=wnid, path=path,
                    video=EncodedImage.from_file(buffer))   # TODO: video
