#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

from .imagenetc import ImageNetC

__all__ = ['ImageNetC']

class_dict: dict[str, ImageSet] = {
    'imagenetc': ImageNetC,
}
