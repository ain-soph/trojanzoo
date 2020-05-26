# -*- coding: utf-8 -*-

import os
from .param import Module
from .model import split_name

from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
# from trojanzoo.attack import Attack

from trojanzoo.config import Config
config = Config.config


def get_module(module_class: str, module_name: str, **kwargs):
    pkg = __import__('trojanzoo.'+module_class, fromlist=['class_dict'])
    class_dict = getattr(pkg, 'class_dict')
    class_name = class_dict[module_name]
    _class = getattr(pkg, class_name)
    return _class(**kwargs)


def get_dataset(module_name: str = None, batch_size: int = None, **kwargs) -> Dataset:
    if module_name is None:
        module_name = config['dataset']['default_dataset']
    if batch_size is None:
        batch_size = config['dataset']['batch_size'][module_name]

    result = Module(config['dataset'])
    result.__delattr__('default_dataset')
    result.__delattr__('batch_size')
    result.update(kwargs)

    return get_module('dataset', module_name, batch_size=batch_size, **result)


def get_model(module_name: str = None, **kwargs) -> Model:
    if module_name is None:
        dataset = 'default'
        if 'dataset' in kwargs.keys():
            dataset = kwargs['dataset'].name
        module_name = config['model']['default_model'][dataset]
    layer = kwargs['layer'] if 'layer' in kwargs.keys() else None
    module_name, layer = split_name(module_name, layer=layer)
    if layer is not None:
        kwargs['layer'] = layer
    return get_module('model', module_name, **kwargs)


def get_attack(module_name, **kwargs):
    return get_module('attack', module_name, **kwargs)
