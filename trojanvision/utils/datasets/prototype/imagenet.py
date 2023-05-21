#!/usr/bin/env python3

from .utils import _Resource

from torchdata.datapipes.iter import Filter, IterDataPipe, Mapper, TarArchiveLoader
from torchvision.prototype.datasets.utils import Dataset, EncodedImage, GDriveResource, OnlineResource
from torchvision.prototype.datapoints import Label
from torchvision.prototype.datasets.utils._internal import (
    hint_sharding,
    hint_shuffling,
)
from torchvision.prototype.datasets._builtin.imagenet import _info  # TODO: unstable

import pathlib
import re
from functools import cached_property
from typing import Any, BinaryIO, cast, Generic, Literal, Match, Sequence, TypeVar


__all__ = ['ImageNetC', 'ImageNetP']


CDistortionCategoryType = Literal['noise', 'blur', 'weather', 'digital', 'extra']
PDistortionCategoryType = Literal['noise', 'blur', 'weather', 'digital', 'validation',
                                  'harder_noise', 'harder_noise_validation']
# DistortionCategoryType = TypeVar('DistortionCategoryType', bound=str)
DistortionCategoryType = TypeVar('DistortionCategoryType', CDistortionCategoryType, PDistortionCategoryType)


class _ImageNetBase(Dataset, Generic[DistortionCategoryType]):
    _DISTORTIONS: dict[DistortionCategoryType, list[str]]
    _RESOURCES: dict[DistortionCategoryType, _Resource]
    _TRAIN_IMAGE_NAME_PATTERN = re.compile(r"(?P<wnid>n\d{8})_\d+[.]JPEG")

    @cached_property
    def distortion_map(self) -> dict[str, DistortionCategoryType]:
        return {distort: k for k, v in self._DISTORTIONS.items() for distort in v}

    def __init__(self, root: str | pathlib.Path,
                 distortion_name: str,
                 severity: Sequence[int] = (1, 2, 3, 4, 5),
                 skip_integrity_check: bool = False) -> None:
        self.distortion_name = distortion_name
        self.severity = severity
        info = _info()
        categories, wnids = info["categories"], info["wnids"]
        self._categories = categories
        self._wnids = wnids
        self._wnid_to_category = dict(zip(wnids, categories))
        super().__init__(root=root, skip_integrity_checkskip_integrity_check=skip_integrity_check)

    def _resources(self) -> list[OnlineResource]:
        distortion_category = self.distortion_map[self.distortion_name]
        return [GDriveResource(**self._RESOURCES[distortion_category])]

    def _datapipe(self, resource_dps: list[IterDataPipe]) -> IterDataPipe[dict[str, Any]]:
        dp = resource_dps[0]
        dp = TarArchiveLoader(dp)
        dp = Filter(dp, lambda x: x.name.endswith('.JPEG'))
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def _prepare_sample(
        self,
        data: tuple[str, BinaryIO],
    ) -> dict[str, Any]:
        path, buffer = data
        wnid = cast(Match[str], self._TRAIN_IMAGE_NAME_PATTERN.match(pathlib.Path(path).name))["wnid"]
        label = Label.from_category(self._wnid_to_category[wnid], categories=self._categories)
        return dict(label=label, wnid=wnid, path=path,
                    image=EncodedImage.from_file(buffer))

    def __len__(self) -> int:
        return 10_000


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
