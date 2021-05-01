#!/usr/bin/env python3

from trojanzoo.configs import config, Config
from trojanzoo.datasets import Dataset
from trojanzoo.models import Model
from trojanzoo.attacks import Attack
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi
from trojanzoo.utils.process import Model_Process

import os
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from typing import Union    # TODO: python 3.10
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    pass


class Defense(ABC, Model_Process):
    name: str = 'defense'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        group.add_argument('--defense', dest='defense_name')
        group.add_argument('--defense_dir', help='directory to contain defense results')
        return group

    def __init__(self, attack: Attack = None, **kwargs):
        super().__init__(**kwargs)
        self.attack = attack

    @abstractmethod
    def detect(self, *args, **kwargs):
        ...


def add_argument(parser: argparse.ArgumentParser, defense_name: str = None, defense: Union[str, Defense] = None,
                 class_dict: dict[str, type[Defense]] = {}):
    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    group = parser.add_argument_group('{yellow}defense{reset}'.format(**ansi), description=defense_name)
    try:
        DefenseType = class_dict[defense_name]
    except KeyError as e:
        if defense_name is None:
            print('{red}you need to first claim the defense name using "--defense".{reset}'.format(**ansi))
        print(f'{defense_name} not in \n{list(class_dict.keys())}')
        raise e
    return DefenseType.add_argument(group)


def create(defense_name: str = None, defense: Union[str, Defense] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           model_name: str = None, model: Union[str, Model] = None,
           config: Config = config, class_dict: dict[str, type[Defense]] = {}, **kwargs):
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    general_config = config.get_config(dataset_name=dataset_name)['defense']
    specific_config = config.get_config(dataset_name=dataset_name)[defense_name]
    result = general_config.update(specific_config).update(kwargs)    # TODO: linting issues
    try:
        DefenseType = class_dict[defense_name]
    except KeyError as e:
        print(f'{defense_name} not in \n{list(class_dict.keys())}')
        raise e
    if folder_path is None:
        folder_path = result['defense_dir']
        if isinstance(dataset, Dataset):
            folder_path = os.path.join(folder_path, dataset.data_type, dataset.name)
        if model_name is not None:
            folder_path = os.path.join(folder_path, model_name)
        folder_path = os.path.join(folder_path, DefenseType.name)
    return DefenseType(name=defense_name, dataset=dataset, model=model, folder_path=folder_path, **result)
