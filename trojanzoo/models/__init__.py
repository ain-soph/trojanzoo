# -*- coding: utf-8 -*-

from .model import Model
from .imagemodel import ImageModel
from .image import *
from trojanzoo.datasets.dataset import Dataset
from trojanzoo.utils.model import split_name
from trojanzoo.utils.config import Config
from trojanzoo.utils.output import ansi

import argparse
import sys


class_dict = {
    'net': Net,
    'alexnet': AlexNet,
    'resnet': ResNet,
    'resnetcomp': ResNetcomp,
    'vgg': VGG,
    'vggcomp': VGGcomp,
    'densenet': DenseNet,
    'densenetcomp': DenseNetcomp,
    'magnet': MagNet,
}


def register(name: str, _class: type):
    class_dict[name] = _class


def add_argument(parser: argparse.ArgumentParser, model_name: str = None) -> argparse._ArgumentGroup:
    if model_name is None:
        model_name = get_model_name()
    ModelType = Model
    if model_name is not None:
        model_name, layer = split_name(model_name)
        ModelType: type[Model] = class_dict[model_name]
    group = parser.add_argument_group('{yellow}model{reset}'.format(**ansi), description=model_name)
    ModelType.add_argument(group)
    return group


def create(model_name: str = None, dataset_name: str = None, dataset: Dataset = None, layer: int = None, **kwargs) -> Model:
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    if model_name is None:
        model_name: str = Config.config['model']['default_model'][dataset_name]
    model_name, layer = split_name(model_name, layer=layer)
    result = Config.combine_param(config=Config.config['model'], dataset_name=dataset_name, **kwargs)
    ModelType: type[Model] = class_dict[model_name]
    return ModelType(dataset=dataset, layer=layer, **result)


def get_model_name() -> str:
    argv = sys.argv
    try:
        idx = argv.index('--model')
        model_name: str = argv[idx + 1]
    except ValueError as e:
        try:
            idx = argv.index('-m')
            model_name: str = argv[idx + 1]
        except ValueError as e:
            return None
    return model_name
