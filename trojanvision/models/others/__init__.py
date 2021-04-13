#!/usr/bin/env python3

from .magnet import MagNet
from trojanvision.models.imagemodel import ImageModel

__all__ = ['MagNet']

class_dict: dict[str, type[ImageModel]] = {
    'magnet': MagNet,
}
