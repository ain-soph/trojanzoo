#!/usr/bin/env python3

from .imagemodel import ImageModel
from .nas import *

from .alexnet import AlexNet
from .bit import BiT
from .densenet import DenseNet
from .dla import DLA
from .magnet import MagNet
from .mobilenet import MobileNet
from .net import Net
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .vgg import VGG
from trojanvision.datasets import ImageSet
from trojanvision.configs import Config, config
import trojanzoo.models
from trojanzoo.utils import get_name

import argparse
import os
from typing import Union

class_dict: dict[str, type[ImageModel]] = {
    'alexnet': AlexNet,
    'bit': BiT,
    'densenet': DenseNet,
    'dla': DLA,
    'magnet': MagNet,
    'mobilenet': MobileNet,
    'net': Net,
    'resnet': ResNet,
    'shufflenetv2': ShuffleNetV2,
    'vgg': VGG,

    'natsbench': NATSbench,
    'darts': DARTS,
    'enas': ENAS,
    'lanet': LaNet,
    'mnasnet': MNASNet,
    'pnasnet': PNASNet,
    'proxylessnas': ProxylessNAS,
}


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: Union[str, ImageModel] = None,
                 config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict) -> argparse._ArgumentGroup:
    dataset_name = get_name(arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']
    model_name = get_model_class(model_name, class_dict=class_dict)
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

    ModelType: type[ImageModel] = class_dict[get_model_class(model_name, class_dict=class_dict)]
    if folder_path is None and isinstance(dataset, ImageSet):
        folder_path = os.path.join(result['model_dir'], dataset.data_type, dataset.name)
    return ModelType(name=model_name, dataset=dataset, folder_path=folder_path, **result)


def get_model_class(name: str, class_dict: dict[str, type[ImageModel]] = class_dict) -> str:
    for class_name in class_dict.keys():
        if class_name in name.lower():
            return class_name
    raise KeyError(f'{class_name} not in {list(class_dict.keys())}')
