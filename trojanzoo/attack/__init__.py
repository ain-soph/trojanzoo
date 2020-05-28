# -*- coding: utf-8 -*-

from .adv import *
from .poison import *
from .backdoor import *
from .other import *

class_dict = {
    'attack': 'Attack',
    'adv': 'Adv',
    'backdoor': 'Backdoor',

    'pgd': 'PGD',
    'inference': 'Inference',

    'poison': 'Poison',

    'watermark': 'Watermark',

    'unify': 'Unify',
}
