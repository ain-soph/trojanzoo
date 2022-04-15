#!/usr/bin/env python3

from ...abstract import CleanLabelBackdoor

from .clean_label import CleanLabel
from .invisible_poison import InvisiblePoison
from .refool import Refool


__all__ = ['CleanLabel', 'InvisiblePoison', 'Refool']

class_dict: dict[str, type[CleanLabelBackdoor]] = {
    'clean_label': CleanLabel,
    'invisible_poison': InvisiblePoison,
    'refool': Refool,
}
