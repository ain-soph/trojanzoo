#!/usr/bin/env python3

from ..abstract import BackdoorDefense

from .attack_agnostic import *
from .input_filtering import *
from .model_inspection import *
from .training_filtering import *

from . import attack_agnostic, input_filtering, model_inspection, training_filtering

module_list = [attack_agnostic, input_filtering, model_inspection, training_filtering]
__all__ = []
class_dict: dict[str, type[BackdoorDefense]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)
