#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanzoo.datasets import Dataset
from trojanzoo.models import Model
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi
from trojanzoo.utils.process import Model_Process

import torch
import argparse
import os
from typing import Union


class Attack(Model_Process):
    name: str = None

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('--attack', dest='attack_name')
        group.add_argument('--attack_dir', dest='attack_dir',
                           help='directory to contain attack results')
        group.add_argument('--output', dest='output', type=int,
                           help='output level, defaults to 0.')

    def attack(self, **kwargs):
        pass
    # ----------------------Utility----------------------------------- #

    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.Tensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)


def add_argument(parser: argparse.ArgumentParser, attack_name: str = None, attack: Union[str, Attack] = None,
                 class_dict: dict[str, type[Attack]] = None) -> argparse._ArgumentGroup:
    attack_name = get_name(name=attack_name, module=attack, arg_list=['--attack'])
    group = parser.add_argument_group('{yellow}attack{reset}'.format(**ansi), description=attack_name)
    AttackType = class_dict[attack_name]
    return AttackType.add_argument(group)     # TODO: Linting problem


def create(attack_name: str = None, attack: Union[str, Attack] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           model_name: str = None, model: Union[str, Model] = None,
           config: Config = config, class_dict: dict[str, type[Attack]] = {}, **kwargs) -> Attack:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    attack_name = get_name(name=attack_name, module=attack, arg_list=['--attack'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    general_config = config.get_config(dataset_name=dataset_name)['attack']
    specific_config = config.get_config(dataset_name=dataset_name)[attack_name]
    result = general_config._update(specific_config)._update(kwargs)    # TODO: linting issues

    AttackType: type[Attack] = class_dict[attack_name]
    if folder_path is None:
        folder_path = result['attack_dir']
        if isinstance(dataset, Dataset):
            folder_path = os.path.join(folder_path, dataset.data_type, dataset.name)
        if model_name is not None:
            folder_path = os.path.join(folder_path, model_name)
        folder_path = os.path.join(folder_path, AttackType.name)
    return AttackType(name=attack_name, dataset=dataset, model=model, folder_path=folder_path, **result)
