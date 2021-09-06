#!/usr/bin/env python3

from trojanzoo.models import Model
from trojanzoo.configs import config
from trojanzoo.utils import get_name
from trojanzoo.utils.output import ansi, prints
from trojanzoo.utils.fim import KFAC

from typing import TYPE_CHECKING
from typing import Any
from trojanzoo.configs import Config    # TODO: python 3.10
from trojanzoo.datasets import Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import argparse
if TYPE_CHECKING:
    pass


class Trainer:
    name = 'trainer'
    param_list = ['optim_args', 'train_args', 'writer_args',
                  'optimizer', 'lr_scheduler', 'kfac', 'writer']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        group.add_argument('--epoch', type=int, help='training epochs, defaults to config[train][epoch].')
        group.add_argument('--resume', type=int, help='resume training from certain epoch, defaults to be 0.')
        group.add_argument('--lr', type=float, help='learning rate, defaults to 0.1.')
        group.add_argument('--parameters', default='full',
                           help='training parameters ("features", "classifier", "full"), defaults to "full".')
        group.add_argument('--OptimType', help='optimizer type, defaults to SGD.')
        group.add_argument('--momentum', type=float, help='momentum passed to Optimizer, defaults to 0.9.')
        group.add_argument('--weight_decay', type=float, help='weight_decay passed to Optimizer, defaults to 3e-4.')
        group.add_argument('--nesterov', action='store_true', help='enable nesterov for SGD optimizer.')
        group.add_argument('--lr_scheduler', action='store_true', help='enable CosineAnnealingLR scheduler.')
        group.add_argument('--kfac', action='store_true', help='Using KFAC preconditioner.')
        group.add_argument('--amp', action='store_true', help='Automatic Mixed Precision.')
        group.add_argument('--grad_clip', type=float, help='Gradient Clipping max norms.')
        group.add_argument('--validate_interval', type=int,
                           help='validate interval during training epochs, defaults to 10.')
        group.add_argument('--save', action='store_true', help='save training results.')
        group.add_argument('--tensorboard', action='store_true', help='save training logging for tensorboard.')
        group.add_argument('--log_dir', help='save training logging for tensorboard.')
        group.add_argument('--flush_secs', type=int,
                           help='How often, in seconds, to flush the pending events and summaries to disk.')
        return group

    def __init__(self, optim_args: dict[str, Any] = {}, train_args: dict[str, Any] = {},
                 writer_args: dict[str, Any] = {},
                 optimizer: Optimizer = None, lr_scheduler: _LRScheduler = None,
                 kfac: KFAC = None, writer=None):
        self.optim_args = optim_args.copy()   # TODO: issue 6 why? to avoid BadAppend issues
        self.train_args = train_args.copy()
        self.writer_args = writer_args.copy()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.kfac = kfac
        self.writer = writer

    def __getitem__(self, key: str):
        if key in self.train_args.keys():
            return self.train_args[key]
        return getattr(self, key)

    def __getattr__(self, key: str):
        if key in self.train_args.keys():
            return self.train_args[key]
        raise AttributeError(key)

    def keys(self):
        keys = self.param_list.copy()
        keys.remove('optim_args')
        keys.remove('train_args')
        keys.remove('writer_args')
        keys.extend(self.train_args.keys())
        return keys

    def summary(self, indent: int = 0):
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(self.name, **ansi), indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key in self.param_list:
            value = getattr(self, key)
            if value:
                prints('{green}{0:<10s}{reset}'.format(key, **ansi), indent=indent + 10)
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


def add_argument(parser: argparse.ArgumentParser, ClassType: type[Trainer] = Trainer):
    group = parser.add_argument_group('{yellow}trainer{reset}'.format(**ansi))
    return ClassType.add_argument(group)


def create(dataset_name: str = None, dataset: Dataset = None, model: Model = None,
           kfac: bool = False, ClassType: type[Trainer] = Trainer, tensorboard: bool = None,
           config: Config = config, **kwargs):
    assert isinstance(model, Model)
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    result = config.get_config(dataset_name=dataset_name)['trainer'].update(kwargs)

    optim_keys = model.define_optimizer.__code__.co_varnames
    train_keys = model._train.__code__.co_varnames
    optim_args: dict[str, Any] = {}
    train_args: dict[str, Any] = {}
    for key, value in result.items():
        if key in optim_keys:
            _dict = optim_args
        elif key in train_keys and key != 'verbose':
            _dict = train_args
        else:
            continue
        _dict[key] = value
    optimizer, lr_scheduler = model.define_optimizer(T_max=result['epoch'], **optim_args)

    module = model._model
    if optim_args['parameters'] == 'features':
        module = module.features
    elif optim_args['parameters'] == 'classifier':
        module = module.classifier
    kfac_optimizer = None
    if kfac:
        kfac_optimizer = KFAC(module)

    writer = None
    writer_args: dict[str, Any] = {}
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer_keys = SummaryWriter.__init__.__code__.co_varnames   # log_dir, flush_secs, ...
        for key, value in result.items():
            if key in writer_keys:
                writer_args[key] = value
        writer = SummaryWriter(**writer_args)
    return ClassType(optim_args=optim_args, train_args=train_args, writer_args=writer_args,
                     optimizer=optimizer, lr_scheduler=lr_scheduler, kfac=kfac_optimizer, writer=writer)
