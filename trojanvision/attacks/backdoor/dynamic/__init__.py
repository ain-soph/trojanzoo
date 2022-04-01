#!/usr/bin/env python3

from ...abstract import DynamicBackdoor

from .input_aware_dynamic import InputAwareDynamic


__all__ = ['InputAwareDynamic']

class_dict: dict[str, type[DynamicBackdoor]] = {
    'input_aware_dynamic': InputAwareDynamic,
}
