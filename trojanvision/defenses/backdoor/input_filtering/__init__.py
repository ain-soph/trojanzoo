#!/usr/bin/env python3

from ..abstract import InputFiltering
from .neo import NEO
from .strip import STRIP

__all__ = ['NEO', 'STRIP']

class_dict: dict[str, type[InputFiltering]] = {
    'neo': NEO,
    'strip': STRIP,
}
