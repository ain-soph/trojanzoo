#!/usr/bin/env python3

from ...abstract import BackdoorAttack

from .unlearn import Unlearn


__all__ = ['Unlearn']

class_dict: dict[str, type[BackdoorAttack]] = {
    'unlearn': Unlearn,
}
