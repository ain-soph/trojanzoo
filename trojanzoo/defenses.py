#!/usr/bin/env python3

from trojanzoo.configs import config
from trojanzoo.datasets import Dataset
from trojanzoo.models import Model
from trojanzoo.attacks import Attack
from trojanzoo.utils.module import get_name, ModelProcess
from trojanzoo.utils.output import ansi

import os
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from trojanzoo.configs import Config
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    pass


class Defense(ABC, ModelProcess):
    r"""
    | An abstract class representing a defense.
    | It inherits :class:`trojanzoo.utils.module.ModelProcess`.

    Note:
        This is the implementation of defense.
        For users, please use :func:`create` instead, which is more user-friendly.

    Attributes:
        attack (trojanzoo.attacks.Attack | None): The attack instance.
    """
    name: str = 'defense'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        r"""Add defense arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete defense class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
        group.add_argument('--defense', dest='defense_name')
        group.add_argument('--defense_dir',
                           help='directory to contain defense results')
        return group

    def __init__(self, attack: Attack = None, **kwargs):
        super().__init__(**kwargs)
        self.attack = attack

    @abstractmethod
    def detect(self, *args, **kwargs):
        r"""Main detect method (need overriding)."""
        ...


def add_argument(parser: argparse.ArgumentParser,
                 defense_name: None | str = None,
                 defense: None | str | Defense = None,
                 class_dict: dict[str, type[Defense]] = {}):
    r"""
    | Add defense arguments to argument parser.
    | For specific arguments implementation, see :meth:`Defense.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        defense_name (str): The defense name.
        defense (str | Defense): The defense instance or defense name
            (as the alias of `defense_name`).
        class_dict (dict[str, type[Defense]]):
            Map from defense name to defense class.
            Defaults to ``{}``.

    Returns:
        argparse._ArgumentGroup: The argument group.
    """
    defense_name = get_name(
        name=defense_name, module=defense, arg_list=['--defense'])
    group = parser.add_argument_group(
        '{yellow}defense{reset}'.format(**ansi), description=defense_name)
    try:
        DefenseType = class_dict[defense_name]
    except KeyError:
        if defense_name is None:
            print(f'{ansi["red"]}you need to first claim the defense name '
                  f'using "--defense".{ansi["reset"]}')
        print(f'{defense_name} not in \n{list(class_dict.keys())}')
        raise
    return DefenseType.add_argument(group)


def create(defense_name: None | str = None, defense: None | str | Defense = None,
           folder_path: None | str = None,
           dataset_name: str = None, dataset: None | str | Dataset = None,
           model_name: str = None, model: None | str | Model = None,
           config: Config = config, class_dict: dict[str, type[Defense]] = {},
           **kwargs):
    r"""
    | Create a defense instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{defense_dir}/{dataset.data_type}/{dataset.name}/{model.name}/{defense.name}'``.
    | For defense implementation, see :class:`Defense`.

    Args:
        defense_name (str): The defense name.
        defense (str | Defense): The defense instance or defense name
            (as the alias of `defense_name`).
        dataset_name (str): The dataset name.
        dataset (str | trojanzoo.datasets.Dataset):
            Dataset instance or dataset name
            (as the alias of `dataset_name`).
        model_name (str): The model name.
        model (str | Model): The model instance or model name
            (as the alias of `model_name`).
        config (Config): The default parameter config.
        class_dict (dict[str, type[Defense]]):
            Map from defense name to defense class.
            Defaults to ``{}``.
        **kwargs: The keyword arguments
            passed to defense init method.

    Returns:
        Defense: The defense instance.
    """
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model,
                          arg_list=['-m', '--model'])
    defense_name = get_name(
        name=defense_name, module=defense, arg_list=['--defense'])
    if dataset_name is None:
        dataset_name = config.full_config['dataset']['default_dataset']
    general_config = config.get_config(dataset_name=dataset_name)['defense']
    specific_config = config.get_config(
        dataset_name=dataset_name)[defense_name]
    result = general_config.update(specific_config).update(
        kwargs)    # TODO: linting issues
    try:
        DefenseType = class_dict[defense_name]
    except KeyError:
        print(f'{defense_name} not in \n{list(class_dict.keys())}')
        raise
    if 'folder_path' not in result.keys():
        folder_path = result['defense_dir']
        if isinstance(dataset, Dataset):
            folder_path = os.path.join(
                folder_path, dataset.data_type, dataset.name)
        if model_name is not None:
            folder_path = os.path.join(folder_path, model_name)
        folder_path = os.path.join(folder_path, DefenseType.name)
        result['folder_path'] = folder_path
    return DefenseType(name=defense_name, dataset=dataset, model=model, **result)
