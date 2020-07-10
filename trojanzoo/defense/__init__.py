# -*- coding: utf-8 -*-

from .defense import Defense
from .defense_backdoor import Defense_Backdoor
from .adv import *
from .backdoor import *

class_dict = {
    'defense': 'Defense',
    'defense_backdoor': 'Defense_Backdoor',

    'advmind': 'AdvMind',

    'neural_cleanse': 'Neural_Cleanse',
    'strip': 'STRIP',
    'abs': 'ABS',
    'activation_clustering': 'activation_clustering',
    'fine_pruning': 'Fine_Pruning',
    'deepinspect': 'DeepInspect'
}
