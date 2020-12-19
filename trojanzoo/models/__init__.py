# -*- coding: utf-8 -*-

from .model import Model
from trojanzoo.datasets.dataset import Dataset
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi

import argparse
import os
from typing import Union


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: Union[str, Model] = None,
                 config: Config = config, class_dict: dict[str, type[Model]] = None) -> argparse._ArgumentGroup:
    dataset_name = get_name(arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']

    group = parser.add_argument_group('{yellow}model{reset}'.format(**ansi), description=model_name)
    ModelType = class_dict[model_name]
    return ModelType.add_argument(group)     # TODO: Linting problem


def create(model_name: str = None, model: Union[str, Model] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           config: Config = config, class_dict: dict[str, type[Model]] = {}, **kwargs) -> Model:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['model']._update(kwargs)
    model_name = model_name if model_name is not None else result['default_model']

    ModelType: type[Model] = class_dict[model_name]
    if folder_path is None and isinstance(dataset, Dataset):
        folder_path = os.path.join(result['model_dir'], dataset.data_type, dataset.name)
    return ModelType(name=model_name, dataset=dataset, folder_path=folder_path, **result)
