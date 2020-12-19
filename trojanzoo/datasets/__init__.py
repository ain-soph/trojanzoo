# -*- coding: utf-8 -*-

from .dataset import Dataset
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi

import argparse
import os
from typing import Union


def add_argument(parser: argparse.ArgumentParser, dataset_name: str = None, dataset: Union[str, Dataset] = None,
                 config: Config = config, class_dict: dict[str, type[Dataset]] = {}) -> argparse._ArgumentGroup:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None else config.get_full_config()['dataset']['default_dataset']
    group = parser.add_argument_group('{yellow}dataset{reset}'.format(**ansi), description=dataset_name)
    DatasetType = class_dict[dataset_name]
    return DatasetType.add_argument(group)     # TODO: Linting problem


def create(dataset_name: str = None, dataset: str = None, folder_path: str = None,
           config: Config = config, class_dict: dict[str, type[Dataset]] = None, **kwargs) -> Dataset:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None else config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['dataset']._update(kwargs)

    DatasetType = class_dict[dataset_name]
    folder_path = folder_path if folder_path is not None else \
        os.path.join(result['data_dir'], DatasetType.data_type, DatasetType.name)     # TODO: Linting problem
    return DatasetType(folder_path=folder_path, **result)
