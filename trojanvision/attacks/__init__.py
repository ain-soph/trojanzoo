#!/usr/bin/env python3

from trojanzoo.attacks import Attack

from .abstract import *

from .adv import *
from .poison import *
from .backdoor import *

from . import adv, poison, backdoor

from trojanvision.configs import config
import trojanzoo.attacks

import argparse
from trojanvision.datasets import ImageSet
from trojanvision.models import ImageModel
from trojanzoo.configs import Config


module_list = [adv, backdoor, poison]
__all__ = ['add_argument', 'create', 'Attack']
class_dict: dict[str, type[Attack]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)


def add_argument(parser: argparse.ArgumentParser,
                 attack_name: str = None, attack: str | Attack = None,
                 class_dict: dict[str, type[Attack]] = class_dict):
    r"""
    | Add attack arguments to argument parser.
    | For specific arguments implementation, see :func:`trojanzoo.attacks.Attack.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        attack_name (str): The attack name.
        attack (str | Attack): The attack instance or attack name
            (as the alias of `attack_name`).
        class_dict (dict[str, type[Attack]]):
            Map from attack name to attack class.
            Defaults to ``trojanvision.attacks.class_dict``.

    Returns:
        argparse._ArgumentGroup: The argument group.

    See Also:
        :func:`trojanzoo.attacks.add_argument()`
    """
    return trojanzoo.attacks.add_argument(parser=parser, attack_name=attack_name, attack=attack,
                                          class_dict=class_dict)


def create(attack_name: str = None, attack: str | Attack = None,
           dataset_name: str = None, dataset: str | ImageSet = None,
           model_name: str = None, model: str | ImageModel = None,
           config: Config = config, class_dict: dict[str, type[Attack]] = class_dict,
           **kwargs) -> Attack:
    r"""
    | Create an attack instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{attack_dir}/{dataset.data_type}/{dataset.name}/{model.name}/{attack.name}'``.
    | For attack implementation, see :class:`trojanzoo.attacks.Attack`.

    Args:
        attack_name (str): The attack name.
        attack (str | Attack): The attack instance or attack name
            (as the alias of `attack_name`).
        dataset_name (str): The dataset name.
        dataset (str | ImageSet):
            Dataset instance or dataset name
            (as the alias of `dataset_name`).
        model_name (str): The model name.
        model (str | ImageModel): The model instance or model name
            (as the alias of `model_name`).
        config (Config): The default parameter config.
        class_dict (dict[str, type[Attack]]):
            Map from attack name to attack class.
            Defaults to ``trojanvision.attacks.class_dict``.
        **kwargs: Keyword arguments
            passed to attack init method.

    Returns:
        Attack: The attack instance.

    See Also:
        :func:`trojanzoo.attacks.create()`
    """
    return trojanzoo.attacks.create(attack_name=attack_name, attack=attack,
                                    dataset_name=dataset_name, dataset=dataset,
                                    model_name=model_name, model=model,
                                    config=config, class_dict=class_dict,
                                    **kwargs)
