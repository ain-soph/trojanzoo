#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from trojanzoo.datasets import Dataset
from trojanzoo.models import Model
from trojanzoo.configs import config, Config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi, prints

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import argparse


class Trainer:
    name = 'trainer'
    param_list: list[str] = ['optim_args', 'train_args', 'writer_args',
                             'optimizer', 'lr_scheduler', 'writer']

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        group.add_argument('--epoch', dest='epoch', type=int,
                           help='training epochs, defaults to config[train][epoch].')
        group.add_argument('--lr', dest='lr', type=float,
                           help='learning rate, defaults to 0.1.')
        group.add_argument('--parameters', dest='parameters', default='full',
                           help='training parameters (\'features\', \'classifier\', \'full\'), defaults to \'full\'.')
        group.add_argument('--OptimType', dest='OptimType',
                           help='optimizer type, defaults to SGD.')
        group.add_argument('--momentum', dest='momentum', type=float,
                           help='momentum passed to Optimizer, defaults to 0.9.')
        group.add_argument('--weight_decay', dest='weight_decay', type=float,
                           help='momentum passed to Optimizer, defaults to 1e-4.')
        group.add_argument('--nesterov', dest='nesterov', action='store_true',
                           help='use nesterov for SGD.')
        group.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true',
                           help='use torch.optim.lr_scheduler.StepLR.')
        group.add_argument('--lr_step_size', dest='lr_step_size', type=int,
                           help='step_size passed to torch.optim.lr_scheduler.StepLR, defaults to 50.')
        group.add_argument('--lr_gamma', dest='lr_gamma', type=float,
                           help='gamma passed to torch.optim.lr_scheduler.StepLR, defaults to 0.1.')
        group.add_argument('--amp', dest='amp', action='store_true',
                           help='Automatic Mixed Precision.')
        group.add_argument('--validate_interval', dest='validate_interval', type=int,
                           help='validate interval during training epochs, defaults to 10.')
        group.add_argument('--save', dest='save', action='store_true',
                           help='save training results.')
        group.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                           help='save training logging for tensorboard.')
        group.add_argument('--log_dir', dest='log_dir',
                           help='save training logging for tensorboard.')
        group.add_argument('--flush_secs', dest='flush_secs', type=int,
                           help='How often, in seconds, to flush the pending events and summaries to disk.')
        return group

    def __init__(self, optim_args: dict = {}, train_args: dict = {}, writer_args: dict = {},
                 optimizer: Optimizer = None, lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        self.optim_args = {} | optim_args   # to avoid BadAppend issues
        self.train_args = {} | train_args
        self.writer_args = {} | writer_args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer

    def __getitem__(self, key: str):
        if key in self.train_args.keys():
            return self.train_args[key]
        return getattr(self, key)

    def __getattr__(self, key: str):
        if key in self.train_args.keys():
            return self.train_args[key]
        raise AttributeError(key)

    def keys(self) -> list[str]:
        keys: list[str] = self.param_list.copy()
        keys.remove('optim_args')
        keys.remove('train_args')
        keys.remove('writer_args')
        keys.extend(list(self.train_args.keys()))
        return keys

    def summary(self, indent: int = 0):
        prints('{blue_light}{0:<20s}{reset} Parameters: '.format(self.name, **ansi), indent=indent)
        for key in self.param_list:
            value = getattr(self, key)
            if value is not None:
                prints('{green}{0:<10s}{reset}'.format(key, **ansi), indent=indent + 10)
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group('{yellow}trainer{reset}'.format(**ansi))
    return Trainer.add_argument(group)


def create(dataset_name: str = None, dataset: Dataset = None, model: Model = None,
           tensorboard: bool = None, config: Config = config, **kwargs) -> tuple[Optimizer, _LRScheduler, dict]:
    assert isinstance(model, Model)
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    result = config.get_config(dataset_name=dataset_name)['trainer']._update(kwargs)

    optim_keys = model.define_optimizer.__code__.co_varnames
    train_keys = model._train.__code__.co_varnames
    writer_keys = SummaryWriter.__init__.__code__.co_varnames   # log_dir, flush_secs, ...
    optim_args = {}
    train_args = {}
    writer_args = {}
    for key, value in result.items():
        if key in optim_keys:
            _dict = optim_args
        elif key in train_keys:
            _dict = train_args
        elif key in writer_keys:
            _dict = writer_args
        else:
            continue
        _dict[key] = value
    optimizer, lr_scheduler = model.define_optimizer(**optim_args)
    writer = SummaryWriter(**writer_args) if tensorboard else None
    return Trainer(optim_args=optim_args, train_args=train_args,
                   optimizer=optimizer, lr_scheduler=lr_scheduler,
                   writer=writer)
