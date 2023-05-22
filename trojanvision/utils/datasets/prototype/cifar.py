#!/usr/bin/env python3

from .utils import _Resource

import torch
from torchvision.datapoints import Image
from torchvision.prototype.datapoints import Label
from torchvision.prototype.datasets.utils._internal import hint_sharding, hint_shuffling
from torchvision.prototype.datasets.utils import Dataset, GDriveResource, OnlineResource
from torchvision.prototype.datasets import Cifar10, Cifar100
from torchdata.datapipes.iter import IterDataPipe, Mapper, Zipper

import numpy as np
import pathlib
from abc import ABC
from typing import Any, Literal, BinaryIO, Generic, Iterator, TypeVar


__all__ = ['Cifar10C', 'Cifar100C', 'Cifar10P', 'Cifar100P']


CDistortionType = Literal['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
                          'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
                          'pixelate', 'shot_noise', 'snow', 'zoom_blur',
                          # extra
                          'gaussian_blur', 'saturate', 'spatter', 'speckle_noise',
                          ]
PDistortionType = Literal['brightness', 'gaussian_noise', 'motion_blur', 'rotate', 'scale',
                          'shot_noise', 'snow', 'tilt', 'translate', 'zoom_blur',
                          # extra
                          'gaussian_blur', 'gaussian_blur_2', 'gaussian_blur_3',
                          'shear', 'shot_noise_2', 'shot_noise_3', 'spatter',
                          'speckle_noise', 'speckle_noise_2', 'speckle_noise_3',
                          ]
# DistortionType = TypeVar('DistortionType', bound=str)
DistortionType = TypeVar('DistortionType', CDistortionType, PDistortionType)


class CifarFileReader(IterDataPipe[torch.Tensor]):
    def __init__(self, datapipe: IterDataPipe[tuple[Any, BinaryIO]]):
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _, file in self.datapipe:
            data = torch.from_numpy(np.load(file))
            file.close()
            yield from data


class _CifarBase(Dataset, Generic[DistortionType], ABC):
    _categories: list[str]
    _RESOURCES: dict[str, _Resource]

    def __init__(self, root: str | pathlib.Path,
                 distortion_name: DistortionType,
                 skip_integrity_check: bool = False) -> None:
        self.distortion_name = distortion_name
        super().__init__(root=root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> list[OnlineResource]:
        return [GDriveResource(**self._RESOURCES[self.distortion_name]),
                GDriveResource(**self._RESOURCES['labels'])]

    def _datapipe(self, resource_dps: list[IterDataPipe]) -> IterDataPipe[dict[str, Any]]:
        images_dp, labels_dp = resource_dps
        images_dp = CifarFileReader(images_dp)
        labels_dp = CifarFileReader(labels_dp)
        dp = Zipper(images_dp, labels_dp)
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        return Mapper(dp, self._prepare_sample)

    def _prepare_sample(self, data: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        image, label = data
        return dict(
            image=Image(image),
            label=Label(label, dtype=torch.int64, categories=self._categories),
        )

    def __len__(self) -> int:
        return 10_000


class _Cifar10Base:
    _categories = Cifar10._categories


class _Cifar100Base:
    _categories = Cifar100._categories


class Cifar10C(_CifarBase[CDistortionType], _Cifar10Base):
    _RESOURCES = {
        'labels': dict(
            id='14ZZ_BHJNBZYNnduwJ15MrJL46BE_O8ON',
            file_name='labels.npy',
            sha256='e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b',
        ),
        'brightness': dict(
            id='1WlTC89wPY6J27TUpyMvRxIA3T1hMH6Qu',
            file_name='brightness.npy',
        ),
        'contrast': dict(
            id='1To61fGSq1SanQ5A0KmmP_zvkKRBqIMS7',
            file_name='contrast.npy',
        ),
        'defocus_blur': dict(
            id='1ONmYTrcZTr_tMgK1dJ9AKn7rs1iRHgMI',
            file_name='defocus_blur.npy',
        ),
        'elastic_transform': dict(
            id='1Xk_j7jzjWD5Pg3IycenbjINKtxftaL-',
            file_name='elastic_transform.npy',
        ),
        'fog': dict(
            id='12D7pnK0G-eAHvRiN2VY-ey8Iaf7XJUJK',
            file_name='fog.npy',
        ),
        'frost': dict(
            id='1k0Ks_6CI2ZloT4Rf2oaJno736mM2mzPY',
            file_name='frost.npy',
        ),
        'gaussian_noise': dict(
            id='1rK851Q070VNff8QH3lhtZiHR4A5MQbS7',
            file_name='gaussian_noise.npy',
        ),
        'glass_blur': dict(
            id='1Wtt97pE1VJZ2ed4LhDmf3a2tE87odoOg',
            file_name='glass_blur.npy',
        ),
        'impulse_noise': dict(
            id='1xZn7LwRDHxG9Gpjm_47kU4DOeu9_t5EQ',
            file_name='impulse_noise.npy',
        ),
        'jpeg_compression': dict(
            id='1vQmJY1azMDwJC5KMXrsv9saw5tFI8dYd',
            file_name='jpeg_compression.npy',
        ),
        'motion_blur': dict(
            id='1XnMqfKVsX0WaJqPwo6SoGzXN0v4okAVT',
            file_name='motion_blur.npy',
        ),
        'pixelate': dict(
            id='1kcaPndFYSj1diQsgcGvSXqWzVZg_2XN-',
            file_name='pixelate.npy',
        ),
        'shot_noise': dict(
            id='1dqjmGqmB3G-MOc20qEVnPygN9aP7SubQ',
            file_name='shot_noise.npy',
        ),
        'snow': dict(
            id='1eJA5vRErfToTK51q6cJDf4wcE7cSFPQ-',
            file_name='snow.npy',
            sha256='84d3fdf8b17e18b7657804321dc13dd260ea1151c9bc91e55e13656e5f5410f2',
        ),
        'zoom_blur': dict(
            id='1o89SAi8tBi2yjQo0uebjLjWjZkTl_sfY',
            file_name='zoom_blur.npy',
        ),
        # extra
        'gaussian_blur': dict(
            id='1C6FF6w5SXsq7G8xB9riXwGh8quhzppO5',
            file_name='gaussian_blur.npy',
        ),
        'saturate': dict(
            id='1hiqADw9puknyw_SETvdz4yiWtr-_UbYZ',
            file_name='saturate.npy',
        ),
        'spatter': dict(
            id='1sUIWx5UxQCeDeD8gwuVzzDDkoFfcgcuL',
            file_name='spatter.npy',
        ),
        'speckle_noise': dict(
            id='1hhkFa76daDt_UF5pft70oD_zlxexYlje',
            file_name='speckle_noise.npy',
        ),
    }


class Cifar100C(_CifarBase[CDistortionType], _Cifar100Base):
    _RESOURCES = {
        'labels': dict(
            id='195yIHzCLUZg6oyDey5-5O2ylnVyS8029',
            file_name='labels.npy',
        ),
        'brightness': dict(
            id='1-PJCpz6Q1izeP1_eWhd4cFoAzd3944tU',
            file_name='brightness.npy',
        ),
        'contrast': dict(
            id='1T48sXevMmO76DOoB1Dr3F8UWLkratKCO',
            file_name='contrast.npy',
        ),
        'defocus_blur': dict(
            id='19AoAuD4MRhgkNGlmPx5G6wZs3bw27ibg',
            file_name='defocus_blur.npy',
        ),
        'elastic_transform': dict(
            id='1v63vH7i4by2G8hACJMfgBYFfNNjqBA7E',
            file_name='elastic_transform.npy',
        ),
        'fog': dict(
            id='1F5A6opzHYcb_K-psANCwcr4qnYVOF9Z_',
            file_name='fog.npy',
        ),
        'frost': dict(
            id='1j7lGf6BwnHcvQY0LFur64QKZZi1M0Rna',
            file_name='frost.npy',
        ),
        'gaussian_noise': dict(
            id='1_2pwfkknFarEfwVNAFVheuCfsSur2E08',
            file_name='gaussian_noise.npy',
        ),
        'glass_blur': dict(
            id='1WAVfitIDXn5HL861btrHuv7TH7eHeoPQ',
            file_name='glass_blur.npy',
        ),
        'impulse_noise': dict(
            id='12E2QrPxRAB0Isov-iIUawYI2Z45p4C12',
            file_name='impulse_noise.npy',
        ),
        'jpeg_compression': dict(
            id='1d0w7Ni7e2qe2OMNCY1ohjt2kPnkv9bJc',
            file_name='jpeg_compression.npy',
        ),
        'motion_blur': dict(
            id='1xOImUI84DKdtFWp5ziSoznwT04ArjE4v',
            file_name='motion_blur.npy',
        ),
        'pixelate': dict(
            id='1cZoDdZbhfEXMQOHGeb0FLgpQrLPYGuM5',
            file_name='pixelate.npy',
        ),
        'shot_noise': dict(
            id='1L0nImNUsld7frTSAipzCBUcGH466aJxU',
            file_name='shot_noise.npy',
        ),
        'snow': dict(
            id='1w92VoRPHEJx36_Y1JNrTYBMquDcAzMZx',
            file_name='snow.npy',
        ),
        'zoom_blur': dict(
            id='1x35UPyVohArUBxBI76oM2_aFb2cU3iz5',
            file_name='zoom_blur.npy',
        ),
        # extra
        'gaussian_blur': dict(
            id='1NhDP5sP4x_DEcpN6iCbXwcblWAixUJfU',
            file_name='gaussian_blur.npy',
        ),
        'saturate': dict(
            id='1tmgKLqo3EXNKTBk1IXKYB58d5Jk7yaBZ',
            file_name='saturate.npy',
        ),
        'spatter': dict(
            id='1Nl9GjunEB_B89cUxbqqbAhhzWUOFadgQ',
            file_name='spatter.npy',
        ),
        'speckle_noise': dict(
            id='16jsnmYUbL7wfukz-MvO_phGUm8srLNI2',
            file_name='speckle_noise.npy',
        ),
    }


class Cifar10P(_CifarBase[PDistortionType], _Cifar10Base):
    _RESOURCES = {
        'labels': dict(
            id='14ZZ_BHJNBZYNnduwJ15MrJL46BE_O8ON',
            file_name='labels.npy',
            sha256='e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b',
        ),
        'brightness': dict(
            id='1Kq_215neP93GeYGZ_bw2xyMq-nhxc_Sw',
            file_name='brightness.npy',
        ),
        'gaussian_noise': dict(
            id='1Y_cv_C2IQXnxZtVIyPRJKiQ6Y-Wsq3cy',
            file_name='gaussian_noise.npy',
        ),
        'motion_blur': dict(
            id='1v2qNl6-cM90tTqneLoNrrY3CrlDjyrjy',
            file_name='motion_blur.npy',
        ),
        'rotate': dict(
            id='14a5zaY4J57VCU-rV2whdHvL58dYdEHaB',
            file_name='rotate.npy',
        ),
        'scale': dict(
            id='1RIdDhh7mCZLcs96zFmI_IhV9V7FrESGJ',
            file_name='scale.npy',
        ),
        'shot_noise': dict(
            id='1DqJ5JcVlPeesuEjLLK_pGM09kXQRUU6O',
            file_name='shot_noise.npy',
        ),
        'snow': dict(
            id='19KN54jNvoOaqC6_Fxth0zNlPV8yJgykz',
            file_name='snow.npy',
        ),
        'tilt': dict(
            id='12UtEWA8BhYTslsOmJ6oj19jeJ4vm49OE',
            file_name='tilt.npy',
        ),
        'translate': dict(
            id='1NGR4RzkLkfroSq6MrFB6g0_LOnQryA7P',
            file_name='translate.npy',
        ),
        'zoom_blur': dict(
            id='1ydL1Oc138L3_QMG_imF5pG4mANpOiNUF',
            file_name='zoom_blur.npy',
        ),
        # extra
        'gaussian_blur': dict(
            id='1df4nY3na2-D5H9Knfx2shKvTAhdukfXw',
            file_name='gaussian_blur.npy',
        ),
        'gaussian_noise_2': dict(
            id='1ZydXOHMXcOw1EPBETTXnQyTX6Po6LMVh',
            file_name='gaussian_noise_2.npy',
        ),
        'gaussian_noise_3': dict(
            id='1xT02_SsSsC7Mzw4RmW3tKqp24zBX2NEd',
            file_name='gaussian_noise_3.npy',
        ),
        'shear': dict(
            id='1bW18zMQ3rWZp0FkkN0wod-alPr3uJFjl',
            file_name='shear.npy',
        ),
        'shot_noise_2': dict(
            id='1C_mdyYfQOk4Pm6B81SkVTUoI1oNKNptJ',
            file_name='shot_noise_2.npy',
        ),
        'shot_noise_3': dict(
            id='1_YxrQ7y8Tw-5X3RVm3XzpjqccNQgBg0s',
            file_name='shot_noise_3.npy',
        ),
        'spatter': dict(
            id='1Oa-sHjrn1hMX2KhyOPEqaLpLfnfivRhU',
            file_name='spatter.npy',
        ),
        'speckle_noise': dict(
            id='1JuGoT9HSALF_fg6Ge16o46hLYEOm96Wv',
            file_name='speckle_noise.npy',
        ),
        'speckle_noise_2': dict(
            id='1WYataTqtIoHX7_6Vt7Q19GBUUX5nvauF',
            file_name='speckle_noise_2.npy',
        ),
        'speckle_noise_3': dict(
            id='11gaMVOzvj5BkDT8NLo9UWUkVil30BBG5',
            file_name='speckle_noise_3.npy',
        ),
    }


class Cifar100P(_CifarBase[PDistortionType], _Cifar100Base):
    _RESOURCES = {
        'labels': dict(
            id='195yIHzCLUZg6oyDey5-5O2ylnVyS8029',
            file_name='labels.npy',
        ),
        'brightness': dict(
            id='1L0dXIl-KJ2lGq3tJcIF14QssnGtQKDNP',
            file_name='brightness.npy',
        ),
        'gaussian_noise': dict(
            id='1284JL5Kg1wXLn12IIwQE4nLYglpiyKsy',
            file_name='gaussian_noise.npy',
        ),
        'motion_blur': dict(
            id='1OHcJhWGaBcRjRAWx7F7jGYLo-S-Itryn',
            file_name='motion_blur.npy',
        ),
        'rotate': dict(
            id='1Nil9wBY25KSBP0CwoRzQTxAj8SvN54-f',
            file_name='rotate.npy',
        ),
        'scale': dict(
            id='1F5q2t59Lq8Bxog-X1wYutoOofDf5LVWr',
            file_name='scale.npy',
        ),
        'shot_noise': dict(
            id='1G86-BMVUzJd2ba8l7fYTpwGOKxlbKClZ',
            file_name='shot_noise.npy',
        ),
        'snow': dict(
            id='1L0qwtrNiUdcSdJx5piko5CZsfixf1bay',
            file_name='snow.npy',
        ),
        'tilt': dict(
            id='1Y8rDqcDp8A0tamQiHXYx7ksR0sqvPt2l',
            file_name='tilt.npy',
        ),
        'translate': dict(
            id='1HYvCWYCG-LvBy60mabGvOhvXiqq8oNnq',
            file_name='translate.npy',
        ),
        'zoom_blur': dict(
            id='1Ba-ivPjakQ_Pp5txl1edruENcFU2RdKd',
            file_name='zoom_blur.npy',
        ),
        # extra
        'gaussian_blur': dict(
            id='1uMxkWuoUWwzHRDftMNdJ55yS33VTGXTH',
            file_name='gaussian_blur.npy',
        ),
        'gaussian_noise_2': dict(
            id='1Ms_-nv6_1x2Fm0hlpguu7mEonuf3KLzI',
            file_name='gaussian_noise_2.npy',
        ),
        'gaussian_noise_3': dict(
            id='1BtWGcJD_3hTnPm1NB69nf4hhv1465Jk4',
            file_name='gaussian_noise_3.npy',
        ),
        'shear': dict(
            id='1jCHauBQKZk2z63X7O5h2vrv9KUuEFHYU',
            file_name='shear.npy',
        ),
        'shot_noise_2': dict(
            id='1DTjztiJVedK9wiIiPnsGoxVa2M1NgRwQ',
            file_name='shot_noise_2.npy',
        ),
        'shot_noise_3': dict(
            id='1KWGiUZOoN-Murig6LaEYrVE4YWzzVxCk',
            file_name='shot_noise_3.npy',
        ),
        'spatter': dict(
            id='1Me2QRK7zYag4LCJ4RkyjAs7vFVOfjTtf',
            file_name='spatter.npy',
        ),
        'speckle_noise': dict(
            id='1clBLxmeJqxfKXjD1Vv96pgILFTgeiOs7',
            file_name='speckle_noise.npy',
        ),
        'speckle_noise_2': dict(
            id='1-qUhCX9yXeHAHn2r6kbARpXIX77cYbW1',
            file_name='speckle_noise_2.npy',
        ),
        'speckle_noise_3': dict(
            id='1NUohTLyyDlyIvyPp4yYzILNj6zGtrzXM',
            file_name='speckle_noise_3.npy',
        ),
    }
