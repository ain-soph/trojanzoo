# -*- coding: utf-8 -*-

from .attack import Attack
from .adv import *
from .poison import *
from .backdoor import *
from trojanzoo.datasets.dataset import Dataset
from trojanzoo.utils.config import Config
from trojanzoo.utils.output import ansi

import argparse
import sys
from typing import Type

class_dict = {
    'attack': Attack,

    'pgd': PGD,
    # 'advmind': AdvMind,

    'poison_basic': Poison_Basic,
    'imc_poison': IMC_Poison,

    'badnet': BadNet,
    'trojannn': TrojanNN,
    'latent_backdoor': Latent_Backdoor,
    'imc': IMC,
    'reflection_backdoor': Reflection_Backdoor,
    'bypass_embed': Bypass_Embed,
    'trojannet': TrojanNet,
    'clean_label': Clean_Label,
    'hidden_trigger': Hidden_Trigger,

    'term_study': Term_Study,
    'unlearn': Unlearn,

    'imc_latent': IMC_Latent,
    'imc_advtrain': IMC_AdvTrain,
    'imc_strip': IMC_STRIP,
    'imc_multi': IMC_Multi,
    'imc_magnet': IMC_MagNet,
    'imc_abs': IMC_ABS,
    'imc_adaptive': IMC_Adaptive,
}


def register(name: str, _class: type):
    class_dict[name] = _class


def add_argument(parser: argparse.ArgumentParser, attack_name: str = None) -> argparse._ArgumentGroup:
    if attack_name is None:
        attack_name = get_attack_name()
    AttackType: Type[Attack] = class_dict[attack_name]
    group = parser.add_argument_group('{yellow}attack{reset}'.format(**ansi),
                                      description='{blue_light}{0}{reset}'.format(attack_name, **ansi))
    AttackType.add_argument(group)
    return group


def create(attack_name: str = None, dataset_name: str = None, dataset: Dataset = None, **kwargs) -> Attack:
    if attack_name is None:
        attack_name = get_attack_name()
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    result = Config.combine_param(config=Config.config['attack'], dataset_name=dataset_name)
    specific = Config.combine_param(config=Config.config[attack_name], dataset_name=dataset_name, **kwargs)
    result.update(specific)
    AttackType: Type[Attack] = class_dict[attack_name]
    return AttackType(dataset=dataset, **result)


def get_attack_name() -> str:
    argv = sys.argv
    try:
        idx = argv.index('--attack')
        attack_name: str = argv[idx + 1]
    except ValueError as e:
        print("You need to set '--attack'.")
        raise e
    return attack_name
