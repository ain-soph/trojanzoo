#!/usr/bin/env python3

from .imagenet import _ImageNetBase
from typing import Literal, TypeVar


__all__ = ['ImageNet64C', 'ImageNet64P', 'TinyImageNetC', 'TinyImageNetP']


CDistortionCategoryType = Literal['main', 'extra']
PDistortionCategoryType = Literal['noise', 'blur', 'weather', 'digital', 'validation',
                                  'harder_noise', 'harder_noise_validation']
# DistortionCategoryType = TypeVar('DistortionCategoryType', bound=str)
DistortionCategoryType = TypeVar('DistortionCategoryType', CDistortionCategoryType, PDistortionCategoryType)


class _DownsampleImageNetC(_ImageNetBase[CDistortionCategoryType]):
    _DISTORTIONS = {
        'main': [],
        'extra': [],
    }


class _DownsampleImageNetP(_ImageNetBase[PDistortionCategoryType]):
    _DISTORTIONS = {
        'noise': ['gaussian_noise', 'impulse_noise'],
        'blur': ['motion_blur', 'zoom_blur'],
        'weather': ['brightness', 'snow'],
        'digital': ['translate', 'rotate', 'tilt', 'scale'],
        'validation': ['speckle_noise', 'gaussian_blur', 'spatter', 'shear'],
        # 'harder_noise': ['harder_noise'],
        # 'harder_noise_validation': ['harder_noise_validation'],
    }


class _TinyImageNet:
    def __len__(self) -> int:
        return 10_000


class ImageNet64C(_DownsampleImageNetC):
    _RESOURCES = {
        'main': dict(
            id='18x8idy9RESFkqpgnT46elpRjsGk3Vsg_',
            file_name='main.tar',
        ),
        'extra': dict(
            id='1ehu5KSeibDngSkcTFtN1_9avlN6qom3n',
            file_name='extra.tar',
        ),
    }


class ImageNet64P(_DownsampleImageNetP):
    _RESOURCES = {
        'noise': dict(
            id='181tBUE9RZWFUCYRm2GrBz7BGEnW9fFmA',
            file_name='noise.tar',
        ),
        'blur': dict(
            id='1BglhnU8XEzHB2VK5RJm64AsMgS5qqGYf',
            file_name='blur.tar',
        ),
        'weather': dict(
            id='1NbchhteXFetua2dA91vjoWHjtteB9pwd',
            file_name='weather.tar',
        ),
        'digital': dict(
            id='1Lz5kH4ufoMQoGuY7sLpnVMezZs5QVUo7',
            file_name='digital.tar',
        ),
        'validation': dict(
            id='1XjlmEFAp0CNEOjvmIOwxd4WikA3gpRUe',
            file_name='validation.tar',
        ),
        'harder_noise': dict(
            id='1LcH_k0esNtuYY_ELzMl6g-uWXGvYA5-_',
            file_name='harder_noise.tar',
        ),
        'harder_noise_validation': dict(
            id='1LLqTm-lIrqWN7vhfk9ZT73O8pF6Gt7nt',
            file_name='harder_noise_validation.tar',
        ),
    }


class TinyImageNetC(_DownsampleImageNetC, _TinyImageNet):
    _RESOURCES = {
        'main': dict(
            id='1iclQ0wFBqIs9lFIBDRFWHlLNags8P0_q',
            file_name='main.tar',
        ),
        'extra': dict(
            id='1qVIcHBdCP4Xiossv_LfoAOafC13_AHPw',
            file_name='extra.tar',
        ),
    }


class TinyImageNetP(_DownsampleImageNetP, _TinyImageNet):
    _RESOURCES = {
        'noise': dict(
            id='1NcZBYqNY963vMgyswcd78hnNVh6ztwYx',
            file_name='noise.tar',
        ),
        'blur': dict(
            id='1eZQYjjLgjeSwfTbnjOXktratOevmcyoX',
            file_name='blur.tar',
        ),
        'weather': dict(
            id='1QsA7UPgnuzqB8zFmKFfhztCgau4Marp6',
            file_name='weather.tar',
        ),
        'digital': dict(
            id='1W4clbDYR_G4qdHfaMiu9E0vU34ns0bBc',
            file_name='digital.tar',
        ),
        'validation': dict(
            id='1x7d14mRe456sD0G5QcZdJWatWMY8Bnfg',
            file_name='validation.tar',
        ),
        'harder_noise': dict(
            id='1bq7mv-p_dORgP2mHlN0COojZl9SwxPmI',
            file_name='harder_noise.tar',
        ),
        'harder_noise_validation': dict(
            id='1bXd1_ymIlzCSTRdIsup_0Ddu8LzEOPSH',
            file_name='harder_noise_validation.tar',
        ),
    }
