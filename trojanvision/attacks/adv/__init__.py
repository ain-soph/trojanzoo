#!/usr/bin/env python3

from trojanzoo.attacks import Attack

from .pgd import PGD

__all__ = ['PGD']

class_dict: dict[str, type[Attack]] = {
    'pgd': PGD,
}
