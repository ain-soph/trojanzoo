#!/usr/bin/env python3

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
    'adv_train': AdvTrain,

    # backdoor defense
    # model inspection
    'neural_cleanse': NeuralCleanse,
    'tabor': TABOR,
    'abs': ABS,
    'deep_inspect': DeepInspect,

    # input detection
    'strip': STRIP,
    'neo': NEO,

    # training data inspection
    'activation_clustering': ActivationClustering,
    'spectral_signature': SpectralSignature,
    'neuron_inspect': NeuronInspect,

    # general defense
    'fine_pruning': FinePruning,
    'image_transform': ImageTransform,
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
