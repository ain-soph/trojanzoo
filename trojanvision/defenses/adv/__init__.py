#!/usr/bin/env python3

from trojanzoo.defenses import Defense

from .advmind import AdvMind
from .curvature import Curvature
from .grad_train import GradTrain

__all__ = ['AdvMind', 'Curvature', 'GradTrain']

class_dict: dict[str, type[Defense]] = {
    'advmind': AdvMind,
    'curvature': Curvature,
    'grad_train': GradTrain,
}
