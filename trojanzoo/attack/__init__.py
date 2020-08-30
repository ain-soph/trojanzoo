# -*- coding: utf-8 -*-

from .attack import Attack
from .adv import *
from .poison import *
from .backdoor import *

class_dict = {
    'attack': 'Attack',

    'pgd': 'PGD',
    'inference': 'Inference',

    'poison_basic': 'Poison_Basic',
    'imc_poison': 'IMC_Poison',

    'badnet': 'BadNet',
    'trojannn': 'TrojanNN',
    'hidden_trigger': 'Hidden_Trigger',
    'latent_backdoor': 'Latent_Backdoor',
    'reflection_backdoor': 'Reflection_Backdoor',
    'clean_label': 'Clean_Label',

    'imc': 'IMC',
    'trojannet': 'Trojan_Net',
    'imc_latent': 'IMC_Latent',
}
