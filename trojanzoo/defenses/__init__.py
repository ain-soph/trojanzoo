# -*- coding: utf-8 -*-

from .defense import Defense
from trojanzoo.datasets.dataset import Dataset
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi

import argparse
from typing import Union


def add_argument(parser: argparse.ArgumentParser, defense_name: str = None, defense: Union[str, Defense] = None,
                 class_dict: dict[str, type[Defense]] = None) -> argparse._ArgumentGroup:
    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    group = parser.add_argument_group('{yellow}defense{reset}'.format(**ansi), description=defense_name)
    DefenseType = class_dict[defense_name]
    return DefenseType.add_argument(group)     # TODO: Linting problem


def create(defense_name: str = None, defense: Union[str, Defense] = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           config: Config = config, class_dict: dict[str, type[Defense]] = None, **kwargs) -> Defense:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['model']._update(kwargs)

    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    DefenseType: type[Defense] = class_dict[defense_name]
    return DefenseType(name=defense_name, dataset=dataset, **result)
