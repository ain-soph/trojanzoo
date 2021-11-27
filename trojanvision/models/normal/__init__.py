#!/usr/bin/env python3

from trojanvision.models.imagemodel import ImageModel

from .bit import BiT
from .dla import DLA
from .dpn import DPN
from .net import Net

__all__ = ['BiT', 'DLA', 'DPN', 'Net']

class_dict: dict[str, type[ImageModel]] = {
    'bit': BiT,
    'dla': DLA,
    'dpn': DPN,
    'net': Net,
}
