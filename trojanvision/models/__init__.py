#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imagemodel import ImageModel
from .net import Net
from .alexnet import AlexNet
from .resnet import ResNet, ResNetcomp, ResNetS
from .vgg import VGG, VGGcomp
from .densenet import DenseNet, DenseNetcomp
from .bit import BiT
from .magnet import MagNet
from trojanvision.datasets import ImageSet
from trojanvision.configs import Config, config
import trojanzoo.models
from trojanzoo.utils import get_name

import argparse
import os
import re
from typing import Union

class_dict: dict[str, type[ImageModel]] = {
    'net': Net,
    'alexnet': AlexNet,
    'resnet': ResNet,
    'resnetcomp': ResNetcomp,
    'resnets': ResNetS,
    'vgg': VGG,
    'vggcomp': VGGcomp,
    'densenet': DenseNet,
    'densenetcomp': DenseNetcomp,
    'bit': BiT,
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
    model_name = split_model_name(model_name)
    return trojanzoo.models.add_argument(parser=parser, model_name=model_name, model=model,
                                         config=config, class_dict=class_dict)


def create(model_name: str = None, model: Union[str, ImageModel] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict, **kwargs) -> ImageModel:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']
    result = config.get_config(dataset_name=dataset_name)['model']._update(kwargs)
    model_name = model_name if model_name is not None else result['default_model']

    ModelType: type[ImageModel] = class_dict[split_model_name(model_name)]
    if folder_path is None and isinstance(dataset, ImageSet):
        folder_path = os.path.join(result['model_dir'], dataset.data_type, dataset.name)
    return ModelType(name=model_name, dataset=dataset, folder_path=folder_path, **result)


def split_model_name(name: str) -> str:
    name: str = re.findall(r'[0-9]+|[A-Za-z]+|_', name)[0]
    return name.split('-')[0].lower()
