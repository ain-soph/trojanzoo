#!/usr/bin/env python3

from trojanzoo.datasets import Dataset
from trojanzoo.configs import config
from trojanzoo.utils.module import get_name, ModelProcess
from trojanzoo.utils.output import ansi

import torch
import os

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from trojanzoo.configs import Config
from trojanzoo.models import Model
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    pass


class Attack(ABC, ModelProcess):
    r"""
    | An abstract class representing an attack.
    | It inherits :class:`trojanzoo.utils.module.ModelProcess`.

    Note:
        This is the implementation of attack.
        For users, please use :func:`create` instead, which is more user-friendly.
    """
    name = 'attack'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        r"""Add attack arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete attack class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
        group.add_argument('--attack', dest='attack_name')
        group.add_argument('--output', type=int,
                           help='output level (default: 0)')
        group.add_argument('--attack_dir',
                           help='directory to contain attack results')
        return group

    @abstractmethod
    def attack(self, **kwargs):
        r"""Main attack method (need overriding)."""
        ...

    def generate_target(self, _input: torch.Tensor, idx: int = 1,
                        same: bool = False, **kwargs):
        r"""Generate target labels of a batched input based on
        the classification confidence ranking index.

        Args:
            _input (torch.Tensor): The input tensor.
            idx (int): The classification confidence
                rank of target class.
                Defaults to ``1``.
            same (bool): Generate the same label
                for all samples using mod.
                Defaults to ``False``.

        Returns:
            torch.Tensor:
                The generated target label with shape ``(N)``.

        See Also:
            This method calls
            :meth:`trojanzoo.models.Model.generate_target()`.

            The implementation is in
            :func:`trojanzoo.utils.model.generate_target()`.
        """
        # TODO: issue 1 anyway to avoid coding so many arguments?
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)


def add_argument(parser: argparse.ArgumentParser, attack_name: str = None,
                 attack: str | Attack = None,
                 class_dict: dict[str, type[Attack]] = {}):
    r"""
    | Add attack arguments to argument parser.
    | For specific arguments implementation, see :meth:`Attack.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        attack_name (str): The attack name.
        attack (str | Attack): The attack instance or attack name
            (as the alias of `attack_name`).
        class_dict (dict[str, type[Attack]]):
            Map from attack name to attack class.
            Defaults to ``{}``.

    Returns:
        argparse._ArgumentGroup: The argument group.
    """
    attack_name = get_name(
        name=attack_name, module=attack, arg_list=['--attack'])
    group = parser.add_argument_group(
        '{yellow}attack{reset}'.format(**ansi), description=attack_name)
    try:
        AttackType = class_dict[attack_name]
    except KeyError:
        if attack_name is None:
            print(f'{ansi["red"]}you need to first claim the attack name '
                  f'using "--attack".{ansi["reset"]}')
        print(f'{attack_name} not in \n{list(class_dict.keys())}')
        raise
    return AttackType.add_argument(group)


def create(attack_name: str = None, attack: str | Attack = None,
           dataset_name: str = None, dataset: str | Dataset = None,
           model_name: str = None, model: str | Model = None,
           config: Config = config, class_dict: dict[str, type[Attack]] = {},
           **kwargs) -> Attack:
    r"""
    | Create an attack instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{attack_dir}/{dataset.data_type}/{dataset.name}/{model.name}/{attack.name}'``.
    | For attack implementation, see :class:`Attack`.

    Args:
        attack_name (str): The attack name.
        attack (str | Attack): The attack instance or attack name
            (as the alias of `attack_name`).
        dataset_name (str): The dataset name.
        dataset (str | Dataset):
            Dataset instance or dataset name
            (as the alias of `dataset_name`).
        model_name (str): The model name.
        model (str | Model): The model instance or model name
            (as the alias of `model_name`).
        config (Config): The default parameter config.
        class_dict (dict[str, type[Attack]]):
            Map from attack name to attack class.
            Defaults to ``{}``.
        **kwargs: The keyword arguments
            passed to attack init method.

    Returns:
        Attack: The attack instance.
    """
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model,
                          arg_list=['-m', '--model'])
    attack_name = get_name(
        name=attack_name, module=attack, arg_list=['--attack'])
    if dataset_name is None:
        dataset_name = config.full_config['dataset']['default_dataset']
    general_config = config.get_config(dataset_name=dataset_name)['attack']
    specific_config = config.get_config(dataset_name=dataset_name)[attack_name]
    result = general_config.update(specific_config).update(kwargs)
    try:
        AttackType = class_dict[attack_name]
    except KeyError:
        print(f'{attack_name} not in \n{list(class_dict.keys())}')
        raise
    if 'folder_path' not in result.keys():
        folder_path = result['attack_dir']
        if isinstance(dataset, Dataset):
            folder_path = os.path.join(
                folder_path, dataset.data_type, dataset.name)
        if model_name is not None:
            folder_path = os.path.join(folder_path, model_name)
        folder_path = os.path.join(folder_path, AttackType.name)
        result['folder_path'] = folder_path
    return AttackType(name=attack_name, dataset=dataset, model=model, **result)
