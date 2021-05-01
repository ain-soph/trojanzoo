#!/usr/bin/env python3

from .imagemodel import ImageModel

from .nas import *
from .normal import *
from .others import *

from . import nas, normal, others

from trojanvision.datasets import ImageSet
from trojanvision.configs import Config, config
import trojanzoo.models

import argparse
from typing import Union

module_list = [nas, normal, others]
__all__ = ['ImageModel', 'class_dict', 'add_argument', 'create',
           'get_available_models', 'output_available_models']
class_dict: dict[str, type[ImageModel]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: Union[str, ImageModel] = None,
                 config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict):
    return trojanzoo.models.add_argument(parser=parser, model_name=model_name, model=model,
                                         config=config, class_dict=class_dict)


def create(model_name: str = None, model: Union[str, ImageModel] = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           folder_path: str = None,
           config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict, **kwargs) -> ImageModel:
    return trojanzoo.models.create(model_name=model_name, model=model,
                                   dataset_name=dataset_name, dataset=dataset,
                                   folder_path=folder_path,
                                   config=config, class_dict=class_dict, **kwargs)


def get_available_models(class_dict: dict[str, type[ImageModel]] = class_dict) -> dict[str, list[str]]:
    return trojanzoo.models.get_available_models(class_dict=class_dict)


def output_available_models(class_dict: dict[str, type[ImageModel]] = class_dict, indent: int = 0) -> None:
    return trojanzoo.models.output_available_models(class_dict=class_dict, indent=indent)
