# -*- coding: utf-8 -*-

from .attack import Attack
from trojanzoo.datasets.dataset import Dataset
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi

import argparse
from typing import Union


def add_argument(parser: argparse.ArgumentParser, attack_name: str = None, attack: Union[str, Attack] = None,
                 class_dict: dict[str, type[Attack]] = None) -> argparse._ArgumentGroup:
    attack_name = get_name(name=attack_name, module=attack, arg_list=['--attack'])
    group = parser.add_argument_group('{yellow}attack{reset}'.format(**ansi), description=attack_name)
    AttackType = class_dict[attack_name]
    return AttackType.add_argument(group)     # TODO: Linting problem


def create(attack_name: str = None, attack: Union[str, Attack] = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           config: Config = config, class_dict: dict[str, type[Attack]] = None, **kwargs) -> Attack:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['model']._update(kwargs)

    attack_name = get_name(name=attack_name, module=attack, arg_list=['--attack'])
    AttackType: type[Attack] = class_dict[attack_name]
    return AttackType(name=attack_name, dataset=dataset, **result)
