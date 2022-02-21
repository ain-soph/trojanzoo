#!/usr/bin/env python3

from trojanzoo.attacks import Attack

from .adv import *
from .poison import *
from .backdoor import *

from . import adv, poison, backdoor

from trojanvision.configs import config
import trojanzoo.attacks

import argparse
from trojanvision.datasets import ImageSet
from trojanzoo.configs import Config
from typing import Union


module_list = [adv, backdoor, poison]
__all__ = ['Attack', 'add_argument', 'create']
class_dict: dict[str, type[Attack]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)


def add_argument(parser: argparse.ArgumentParser, attack_name: str = None, attack: Union[str, Attack] = None,
                 class_dict: dict[str, type[Attack]] = class_dict):
    return trojanzoo.attacks.add_argument(parser=parser, attack_name=attack_name, attack=attack,
                                          class_dict=class_dict)


def create(attack_name: str = None, attack: Union[str, Attack] = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, class_dict: dict[str, type[Attack]] = class_dict, **kwargs):
    return trojanzoo.attacks.create(attack_name=attack_name, attack=attack,
                                    dataset_name=dataset_name, dataset=dataset,
                                    config=config, class_dict=class_dict, **kwargs)
