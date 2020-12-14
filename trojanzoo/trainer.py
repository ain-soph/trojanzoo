# -*- coding: utf-8 -*-


from trojanzoo import optim
from trojanzoo.dataset.dataset import Dataset
from trojanzoo.model.model import Model
from trojanzoo.utils.config import Config
from trojanzoo.utils.output import ansi, prints
from trojanzoo.utils.param import Param

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import argparse
from typing import List, Tuple


class Trainer:
    param_list: List[str] = ['optim_args', 'train_args', 'optimizer', 'lr_scheduler']

    def __init__(self, optim_args: dict = {}, train_args: dict = {}, optimizer: Optimizer = None, lr_scheduler: _LRScheduler = None):
        self.optim_args: Param = Param(optim_args)
        self.train_args: Param = Param(train_args)
        self.optimizer: Optimizer = optimizer
        self.lr_scheduler: _LRScheduler = lr_scheduler

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def keys(cls):
        return cls.param_list

    @classmethod
    def summary(cls, indent: int = 0):
        prints('{blue_light}{0:<20s}{reset} Parameters: '.format('train', **ansi), indent=indent)
        for key in cls.param_list:
            value = getattr(cls, key)
            if value is not None:
                prints('{green}{0:<10s}{reset}'.format(key, **ansi), indent=indent + 10)
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group('{yellow}train{reset}'.format(**ansi))
    group.add_argument('--epoch', dest='epoch', type=int,
                       help='training epochs, defaults to config[train][epoch].')
    group.add_argument('--lr', dest='lr', type=float,
                       help='learning rate, defaults to 0.1.')
    group.add_argument('--parameters', dest='parameters', default='full',
                       help='training parameters (\'features\', \'classifier\', \'full\'), defaults to \'full\'.')
    group.add_argument('--optim_type', dest='optim_type',
                       help='optimizer type, defaults to SGD.')
    group.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true',
                       help='use torch.optim.lr_scheduler.StepLR.')
    group.add_argument('--step_size', dest='step_size', type=int,
                       help='step_size passed to torch.optim.lr_scheduler.StepLR, defaults to 50.')
    group.add_argument('--amp', dest='amp', action='store_true',
                       help='Automatic Mixed Precision.')
    group.add_argument('--validate_interval', dest='validate_interval', type=int,
                       help='validate interval during training epochs, defaults to 10.')
    group.add_argument('--save', dest='save', action='store_true',
                       help='save training results.')
    return group


def create(dataset_name: str = None, dataset: Dataset = None, model: Model = None, **kwargs) -> Tuple[Optimizer, _LRScheduler, dict]:
    assert isinstance(model, Model)
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    result = Config.combine_param(config=Config.config['train'], dataset_name=dataset_name, **kwargs)

    func_keys = model.define_optimizer.__code__.co_varnames
    train_keys = model._train.__code__.co_varnames
    optim_args = {}
    train_args = {}
    for key, value in result.items():
        if key in func_keys:
            _dict = optim_args
        elif key in train_keys:
            _dict = train_args
        else:
            continue
        _dict[key] = value

    optimizer, lr_scheduler = model.define_optimizer(**optim_args)
    return Trainer(optim_args=optim_args, train_args=train_args, optimizer=optimizer, lr_scheduler=lr_scheduler)
