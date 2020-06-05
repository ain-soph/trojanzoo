# -*- coding: utf-8 -*-

import os
from .param import Module
from .model import split_name

from trojanzoo.dataset import Dataset, ImageSet
from trojanzoo.model import Model
from trojanzoo.attack import Attack, Watermark

from typing import Dict, List

from trojanzoo.config import Config
config = Config.config


def get_module(module_class: str, module_name: str, **kwargs):
    pkg = __import__('trojanzoo.'+module_class, fromlist=['class_dict'])
    class_dict: Dict[str, str] = getattr(pkg, 'class_dict')
    class_name: str = class_dict[module_name]
    _class = getattr(pkg, class_name)
    return _class(**kwargs)


def get_dataset(module_name: str = None, batch_size: int = None, **kwargs) -> Dataset:
    if module_name is None:
        module_name: str = config['dataset']['default_dataset']
    if batch_size is None:
        batch_size: int = config['dataset']['batch_size'][module_name]

    result: Module = Module(config['dataset'])
    result.__delattr__('default_dataset')
    result.__delattr__('batch_size')
    result.update(kwargs)

    return get_module('dataset', module_name, batch_size=batch_size, **result)


def get_model(module_name: str = None, layer: int = None, dataset: str = None, **kwargs) -> Model:
    if module_name is None:
        if dataset is None:
            dataset: str = 'default'
        elif isinstance(dataset, Dataset):
            dataset = dataset.name
        module_name: str = config['model']['default_model'][dataset]
    module_name, layer = split_name(module_name, layer=layer)

    result: Module = Module(config['model'])
    if layer is not None:
        result.layer = layer
    result.__delattr__('default_model')
    result.update(kwargs)
    return get_module('model', module_name, **result)


def get_attack(module_name, **kwargs):
    return get_module('attack', module_name, **kwargs)


def get_mark(data_shape: List[int] = None, dataset: ImageSet = None, **kwargs):

    if data_shape is None:
        assert isinstance(dataset, Dataset)
        data_shape: list = [dataset.n_channel]
        data_shape.extend(dataset.n_dim)
    result: Module = Module(config['mark'])
    result.data_shape = data_shape
    result.update(kwargs)
    return Watermark(data_shape=data_shape, **result)
