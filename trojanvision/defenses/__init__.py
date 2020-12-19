# -*- coding: utf-8 -*-

from .backdoor_defense import BackdoorDefense
from .adv import *
from .backdoor import *

from trojanvision.configs import Config, config
from trojanvision.datasets import ImageSet
import trojanzoo.defenses
from trojanzoo.defenses import Defense
import argparse
from typing import Union

class_dict = {
    # adversarial Defense
    'advmind': AdvMind,
    'curvature': Curvature,
    # 'grad_train': Grad_Train,
    'adv_train': Adv_Train,

    # backdoor defense
    # model inspection
    'neural_cleanse': NeuralCleanse,
    'tabor': TABOR,
    'abs': ABS,
    'deep_inspect': Deep_Inspect,

    # input detection
    'strip': STRIP,
    'neo': NEO,

    # training data inspection
    'activation_clustering': Activation_Clustering,
    'spectral_signature': Spectral_Signature,
    'neuron_inspect': Neuron_Inspect,

    # general defense
    'fine_pruning': Fine_Pruning,
    'image_transform': Image_Transform,
    'magnet': MagNet,
}


def add_argument(parser: argparse.ArgumentParser, defense_name: str = None, defense: Union[str, Defense] = None,
                 class_dict: dict[str, type[Defense]] = class_dict):
    return trojanzoo.defenses.add_argument(parser=parser, defense_name=defense_name, defense=defense,
                                           class_dict=class_dict)


def create(defense_name: str = None, defense: Union[str, Defense] = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, class_dict: dict[str, type[Defense]] = class_dict, **kwargs):
    return trojanzoo.defenses.create(defense_name=defense_name, defense=defense,
                                     dataset_name=dataset_name, dataset=dataset,
                                     config=config, class_dict=class_dict, **kwargs)
