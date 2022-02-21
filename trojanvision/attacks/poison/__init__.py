#!/usr/bin/env python3

from trojanzoo.attacks import Attack

from .imc_poison import IMC_Poison
from .poison_basic import PoisonBasic
from .poison_random import PoisonRandom

__all__ = ['IMC_Poison', 'PoisonBasic', 'PoisonRandom']

class_dict: dict[str, type[Attack]] = {
    'imc_poison': IMC_Poison,
    'poison_basic': PoisonBasic,
    'poison_random': PoisonRandom,
}
