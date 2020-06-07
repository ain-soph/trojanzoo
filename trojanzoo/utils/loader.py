# -*- coding: utf-8 -*-

import os
from .param import Param
from .model import split_name

from trojanzoo.dataset import Dataset, ImageSet
from trojanzoo.model import Model
from trojanzoo.attack import Attack
from trojanzoo.utils.attack import Watermark

from typing import Dict, List

from trojanzoo.config import Config
config = Config.config


def get_module(module_class: str, module_name: str, **kwargs):
    pkg = __import__('trojanzoo.'+module_class, fromlist=['class_dict'])
    class_dict: Dict[str, str] = getattr(pkg, 'class_dict')
    class_name: str = class_dict[module_name]
    _class = getattr(pkg, class_name)
    return _class(**kwargs)


def get_dataset(module_name: str = None, **kwargs) -> Dataset:
    if module_name is None:
        module_name: str = config['dataset']['default_dataset']
    result: Param = combine_param(config=config['dataset'], sub_idx=module_name,
                                  filter_list=['default_dataset'], **kwargs)
    return get_module('dataset', module_name, **result)


def get_model(module_name: str = None, layer: int = None, dataset: Dataset = None, **kwargs) -> Model:
    if dataset is None:
        dataset_name: str = 'default'
    elif isinstance(dataset, Dataset):
        dataset_name = dataset.name
    if module_name is None:
        module_name: str = config['model']['default_model'][dataset_name]
    module_name, layer = split_name(module_name, layer=layer)

    result: Param = combine_param(config=config['model'], dataset=dataset,
                                  filter_list=['default_model'], layer=layer, **kwargs)
    return get_module('model', module_name, **result)


def get_attack(module_name:str='hello', dataset: Dataset = None, **kwargs):
    result: Param = combine_param(config=config[module_name], dataset=dataset,
                                  **kwargs)
    return get_module('attack', module_name, **kwargs)


def get_mark(data_shape: List[int] = None, dataset: ImageSet = None, **kwargs):
    if data_shape is None:
        assert isinstance(dataset, ImageSet)
        data_shape: list = [dataset.n_channel]
        data_shape.extend(dataset.n_dim)

    result: Param = combine_param(config=config['mark'], dataset=dataset,
                                  data_shape=data_shape, **kwargs)

    return Watermark(**result)


def combine_param(config: Param = None, dataset: Dataset = None, filter_list: List[str] = [], **kwargs):
    if dataset is None:
        dataset_name: str = 'default'
    elif isinstance(dataset, Dataset):
        dataset_name = dataset.name

    result = Param()
    if config:
        result.add(config)
    for key in filter_list:
        if key in result.keys():
            result.__delattr__(key)
    for key, value in result.items():
        if isinstance(value, Param):
            result[key] = value[dataset_name]
    result.update(kwargs)

    if dataset:
        result.dataset: Dataset = dataset
    return result
