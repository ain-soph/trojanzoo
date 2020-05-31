# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.param import Module
from trojanzoo.model import Model
from trojanzoo.config import Config
config = Config.config


class Parser_Train(Parser):

    def __init__(self, name='train'):
        super().__init__(name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--epoch', dest='epoch', type=int)
        parser.add_argument('--lr', dest='lr', type=float)
        parser.add_argument('--parameters', dest='parameters', default='full')
        parser.add_argument('--optim_type', dest='optim_type')
        parser.add_argument('--lr_scheduler', dest='lr_scheduler',
                            action='store_true')
        parser.add_argument('--step_size', dest='step_size', type=int)
        parser.add_argument('--validate_interval', dest='validate_interval', type=int)
        parser.add_argument('--save', dest='save', action='store_true')

    def get_module(self, model: Model, **kwargs):
        dataset = 'default'
        if 'dataset' in kwargs.keys():
            dataset = kwargs['dataset'].name
            kwargs.pop('dataset')

        new_args = Module({key: value[dataset]
                         for key, value in config['train'].items()})
        new_args.update(kwargs)

        func_keys = model.define_optimizer.__code__.co_varnames
        train_keys = model._train.__code__.co_varnames
        optim_args = {}
        train_args = {}
        # other_args = {}
        for key, value in new_args.items():
            if key in func_keys:
                _dict = optim_args
            elif key in train_keys:
                _dict = train_args
            else:
                raise KeyError(key)
                # _dict = other_args
            _dict[key] = value

        optimizer, lr_scheduler = model.define_optimizer(**optim_args)
        return optimizer, lr_scheduler, train_args
