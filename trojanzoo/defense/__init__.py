# -*- coding: utf-8 -*-

from .defense import Defense
from .defense_backdoor import Defense_Backdoor
from .adv import *
from .backdoor import *
from trojanzoo.dataset.dataset import Dataset
from trojanzoo.utils.config import Config
from trojanzoo.utils.output import ansi

import argparse
import sys
from typing import Type

class_dict = {

    'advmind': AdvMind,
    'curvature': Curvature,
    'grad_train': Grad_Train,
    'adv_train': Adv_Train,

    'neural_cleanse': Neural_Cleanse,
    'tabor': TABOR,
    'strip': STRIP,
    'abs': ABS,
    'activation_clustering': Activation_Clustering,
    'fine_pruning': Fine_Pruning,
    'deep_inspect': Deep_Inspect,
    'spectral_signature': Spectral_Signature,
    'neuron_inspect': Neuron_Inspect,
    'image_transform': Image_Transform,
    'magnet': MagNet,
    'neo': NEO,
}


def register(name: str, _class: type, **kwargs):
    class_dict[name] = _class


def add_argument(parser: argparse.ArgumentParser, defense_name: str = None) -> argparse._ArgumentGroup:
    if defense_name is None:
        defense_name = get_defense_name()
    DefenseType: Type[Defense] = class_dict[defense_name]
    group = parser.add_argument_group('{yellow}defense{reset}'.format(**ansi),
                                      description='{blue_light}{0}{reset}'.format(defense_name, **ansi))
    DefenseType.add_argument(group)
    return group


def create(defense_name: str = None, dataset_name: str = None, dataset: Dataset = None, **kwargs) -> Defense:
    if defense_name is None:
        defense_name = get_defense_name()
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    result = Config.combine_param(config=Config.config[defense_name], dataset_name=dataset_name, **kwargs)
    DefenseType: Type[Defense] = class_dict[defense_name]
    return DefenseType(dataset=dataset, **result)


def get_defense_name() -> str:
    argv = sys.argv
    try:
        idx = argv.index('--defense')
        defense_name: str = argv[idx + 1]
    except ValueError as e:
        print("You need to set '--defense'.")
        raise e
    return defense_name
