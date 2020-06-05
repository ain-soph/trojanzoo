# -*- coding: utf-8 -*-

from .attack import Attack
from .backdoor_attack import Backdoor_Attack, Watermark
from .adv import *
from .poison import *
from .backdoor import *
from .other import *

class_dict = {
    'attack': 'Attack',
    'adv_attack': 'Adv_Attack',
    'backdoor_attack': 'Backdoor_Attack',

    'pgd': 'PGD',
    'inference': 'Inference',

    'poison': 'Poison',

    'watermark': 'Watermark',

    'unify': 'Unify',
}
