#!/usr/bin/env python3

from ..abstract import BackdoorAttack

from .normal import *
from .clean_label import *
from .dynamic import *
from .others import *

from . import normal, clean_label, dynamic, others

module_list = [normal, clean_label, dynamic, others]
__all__ = []
class_dict: dict[str, type[BackdoorAttack]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)
