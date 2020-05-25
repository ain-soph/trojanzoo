# -*- coding: utf-8 -*-

from .perturb import Perturb
from .pgd import PGD
from .poison import Poison
from .unify import Unify
# from .inference import Inference
from .watermark import Watermark

class_dict = {
    'perturb': 'Perturb',
    'pgd': 'PGD',
    'poison': 'Poison',
    'unify': 'Unify',
    'inference': 'Inference',
    'watermark': 'Watermark',
}
