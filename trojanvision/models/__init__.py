# -*- coding: utf-8 -*-

from .imagemodel import ImageModel
from .net import Net
from .alexnet import AlexNet
from .resnet import ResNet, ResNetcomp
from .vgg import VGG, VGGcomp
from .densenet import DenseNet, DenseNetcomp
from .magnet import MagNet
from trojanvision.datasets import ImageSet
from trojanvision.utils import split_name
from trojanvision.configs import Config, config

import trojanzoo.models
from trojanzoo.utils import get_name

import argparse
from typing import Union

class_dict: dict[str, type[ImageModel]] = {
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


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: Union[str, ImageModel] = None,
                 config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict) -> argparse._ArgumentGroup:
    dataset_name = get_name(arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']
    model_name, layer = split_name(model_name)
    return trojanzoo.models.add_argument(parser=parser, model_name=model_name, model=model,
                                         config=config, class_dict=class_dict)


def create(model_name: str = None, model: Union[str, ImageModel] = None, layer: int = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict, **kwargs) -> ImageModel:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']
    model_name, layer = split_name(model_name, layer=layer)
    return trojanzoo.models.create(model_name=model_name, model=model,
                                   dataset_name=dataset_name, dataset=dataset,
                                   config=config, class_dict=class_dict,
                                   layer=layer, **kwargs)
