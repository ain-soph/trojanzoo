#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanzoo.configs import config, Config
from trojanzoo.datasets import Dataset
from trojanzoo.models import Model
from trojanzoo.attacks import Attack
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi
from trojanzoo.utils.process import Model_Process

import argparse
import os
from typing import Union


class Defense(Model_Process):

    name: str = None

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('--defense', dest='defense_name')
        group.add_argument('--defense_dir', dest='defense_dir',
                           help='directory to contain defense results')

    def __init__(self, attack: Attack = None, **kwargs):
        super().__init__(**kwargs)
        self.attack: Attack = attack

    def detect(self, **kwargs):
        raise NotImplementedError()


def add_argument(parser: argparse.ArgumentParser, defense_name: str = None, defense: Union[str, Defense] = None,
                 class_dict: dict[str, type[Defense]] = None) -> argparse._ArgumentGroup:
    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    group = parser.add_argument_group('{yellow}defense{reset}'.format(**ansi), description=defense_name)
    DefenseType = class_dict[defense_name]
    return DefenseType.add_argument(group)     # TODO: Linting problem


def create(defense_name: str = None, defense: Union[str, Defense] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           model_name: str = None, model: Union[str, Model] = None,
           config: Config = config, class_dict: dict[str, type[Defense]] = {}, **kwargs) -> Defense:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    defense_name = get_name(name=defense_name, module=defense, arg_list=['--defense'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    general_config = config.get_config(dataset_name=dataset_name)['defense']
    specific_config = config.get_config(dataset_name=dataset_name)[defense_name]
    result = general_config._update(specific_config)._update(kwargs)    # TODO: linting issues

    DefenseType: type[Defense] = class_dict[defense_name]
    if folder_path is None:
        folder_path = result['defense_dir']
        if isinstance(dataset, Dataset):
            folder_path = os.path.join(folder_path, dataset.data_type, dataset.name)
        if model_name is not None:
            folder_path = os.path.join(folder_path, model_name)
        folder_path = os.path.join(folder_path, DefenseType.name)
    return DefenseType(name=defense_name, dataset=dataset, model=model, folder_path=folder_path, **result)
