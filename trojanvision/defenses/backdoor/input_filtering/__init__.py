#!/usr/bin/env python3

from ...abstract import InputFiltering
from .neo import Neo
from .strip import Strip

__all__ = ['Neo', 'Strip']

class_dict: dict[str, type[InputFiltering]] = {
    'neo': Neo,
    'strip': Strip,
}
