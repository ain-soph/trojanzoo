#!/usr/bin/env python3

from .abstract import BackdoorDefense, InputFiltering, TrainingFiltering

from .general import *
from .input_filtering import *
from .model_inspection import *
from .training_filtering import *

from . import general, input_filtering, model_inspection, training_filtering

module_list = [general, input_filtering, model_inspection, training_filtering]
__all__ = ['BackdoorDefense', 'InputFiltering', 'TrainingFiltering']
class_dict: dict[str, type[BackdoorDefense]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)
