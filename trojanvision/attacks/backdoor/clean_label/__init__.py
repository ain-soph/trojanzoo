#!/usr/bin/env python3

from ...abstract import CleanLabelBackdoor

from .invisible_poison import InvisiblePoison
from .refool import Refool


__all__ = ['InvisiblePoison', 'Refool']

class_dict: dict[str, type[CleanLabelBackdoor]] = {
    'invisible_poison': InvisiblePoison,
    'refool': Refool,
}
