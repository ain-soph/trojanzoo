#!/usr/bin/env python3

from trojanzoo.models import Model
from trojanzoo.configs import config
from trojanzoo.environ import env
from trojanzoo.utils.model import ExponentialMovingAverage
from trojanzoo.utils.module import get_name
from trojanzoo.utils.output import ansi, prints
from trojanzoo.utils.fim import KFAC, EKFAC

from typing import TYPE_CHECKING
from typing import Any, Union
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
                  'optimizer', 'lr_scheduler',
                  'pre_conditioner', 'model_ema',
                  'writer']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        group.add_argument('--epochs', type=int,
                           help='training epochs '
                           '(default: config[train][epochs])')
        group.add_argument('--resume', type=int,
                           help='resume training from certain epochs '
                           '(default: 0)')
        group.add_argument('--lr', type=float,
                           help='learning rate (default: 0.1)')
        group.add_argument('--parameters',
                           choices=['features', 'classifier', 'full'],
                           help='training parameters '
                           '("features", "classifier", "full") '
                           '(default: "full")')

        group.add_argument('--OptimType',
                           help='optimizer type (default: SGD)')
        group.add_argument('--momentum', type=float,
                           help='momentum passed to Optimizer (default: 0.9)')
        group.add_argument('--weight_decay', type=float,
                           help='weight_decay passed to Optimizer '
                           '(default: 3e-4)')
        group.add_argument('--nesterov', action='store_true',
                           help='enable nesterov for SGD optimizer')

        group.add_argument('--lr_scheduler', action='store_true',
                           help='enable lr scheduler')
        group.add_argument('--lr_scheduler_type',
                           choices=['StepLR', 'CosineAnnealingLR',
                                    'ExponentialLR'],
                           help='the lr scheduler '
                           '(default: CosineAnnealingLR)')
        group.add_argument('--lr_min', type=float,
                           help='min learning rate for `eta_min` '
                           'in CosineAnnealingLR (default: 0.0)')
        group.add_argument('--lr_warmup_epochs', type=int,
                           help='the number of epochs to warmup (default: 0)')
        group.add_argument('--lr_warmup_method',
                           choices=['constant', 'linear'],
                           help='the warmup method (default: constant)')
        group.add_argument('--lr_step_size', type=int,
                           help='decrease lr every step-size epochs '
                           '(default: 30)')
        group.add_argument('--lr_gamma', type=float,
                           help='decrease lr by a factor of lr-gamma '
                           '(default: 0.1)')

        group.add_argument('--model_ema', action='store_true',
                           help='enable tracking Exponential Moving Average '
                           'of model parameters')
        group.add_argument('--model_ema_steps', type=int,
                           help='the number of iterations that controls '
                           'how often to update the EMA model (default: 32)')
        group.add_argument('--model_ema_decay', type=float,
                           help='decay factor for Exponential Moving Average '
                           'of model parameters (default: 0.99998)')

        group.add_argument('--pre_conditioner', choices=['kfac', 'ekfac'],
                           help='Using kfac/ekfac preconditioner')

        group.add_argument('--amp', action='store_true',
                           help='Use torch.cuda.amp '
                           'for mixed precision training')
        group.add_argument('--grad_clip', type=float,
                           help='Gradient Clipping max norms')

        group.add_argument('--validate_interval', type=int,
                           help='validate interval during training epochs '
                           '(default: 10)')
        group.add_argument('--save', action='store_true',
                           help='save training results')

        group.add_argument('--tensorboard', action='store_true',
                           help='save training logging for tensorboard')
        group.add_argument('--log_dir',
                           help='save training logging for tensorboard')
        group.add_argument('--flush_secs', type=int,
                           help='How often (seconds) '
                           'to flush the pending events '
                           'and summaries to disk')
        return group

    def __init__(self, optim_args: dict[str, Any] = {},
                 train_args: dict[str, Any] = {},
                 writer_args: dict[str, Any] = {},
                 optimizer: Optimizer = None,
                 lr_scheduler: _LRScheduler = None,
                 model_ema: ExponentialMovingAverage = None,
                 pre_conditioner: Union[KFAC, EKFAC] = None, writer=None):
        # TODO: issue 6 why? to avoid BadAppend issues
        self.optim_args = optim_args.copy()
        self.train_args = train_args.copy()
        self.writer_args = writer_args.copy()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_ema = model_ema
        self.pre_conditioner = pre_conditioner
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
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(
            self.name, **ansi),
            indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key in self.param_list:
            value = getattr(self, key)
            if value:
                prints('{green}{0:<10s}{reset}'.format(key, **ansi),
                       indent=indent + 10)
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


def add_argument(parser: argparse.ArgumentParser,
                 ClassType: type[Trainer] = Trainer):
    group = parser.add_argument_group('{yellow}trainer{reset}'.format(**ansi))
    return ClassType.add_argument(group)


def create(dataset_name: str = None,
           dataset: Dataset = None, model: Model = None,
           model_ema: bool = False,
           pre_conditioner: str = None,
           ClassType: type[Trainer] = Trainer,
           tensorboard: bool = None,
           config: Config = config, **kwargs):
    assert isinstance(model, Model)
    dataset_name = get_name(name=dataset_name, module=dataset,
                            arg_list=['-d', '--dataset'])
    result = config.get_config(dataset_name=dataset_name
                               )['trainer'].update(kwargs)

    optim_keys = model.define_optimizer.__code__.co_varnames
    train_keys = model._train.__code__.co_varnames
    optim_args: dict[str, Any] = {}
    train_args: dict[str, Any] = {}
    for key, value in result.items():
        if key in optim_keys:
            optim_args[key] = value
        elif key in train_keys and key != 'verbose':
            train_args[key] = value
    train_args['epochs'] = result['epochs']
    train_args['lr_warmup_epochs'] = result['lr_warmup_epochs']

    optimizer, lr_scheduler = model.define_optimizer(**optim_args)

    module = model._model
    if optim_args['parameters'] == 'features':
        module = module.features
    elif optim_args['parameters'] == 'classifier':
        module = module.classifier

    # https://github.com/pytorch/vision/blob/main/references/classification/train.py
    model_ema_module = None
    if model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = env['world_size'] * dataset.batch_size * \
            result['model_ema_steps'] / result['epochs']
        alpha = 1.0 - result['model_ema_decay']
        alpha = min(1.0, alpha * adjust)
        model_ema_module = ExponentialMovingAverage(
            model._model, decay=1.0 - alpha)

    kfac_optimizer = None
    if pre_conditioner == 'kfac':   # TODO: python 3.10
        kfac_optimizer = KFAC(module)
    elif pre_conditioner == 'ekfac':
        kfac_optimizer = EKFAC(module)

    writer = None
    writer_args: dict[str, Any] = {}
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # log_dir, flush_secs, ...
        writer_keys = SummaryWriter.__init__.__code__.co_varnames
        for key, value in result.items():
            if key in writer_keys:
                writer_args[key] = value
        writer = SummaryWriter(**writer_args)
    return ClassType(optim_args=optim_args, train_args=train_args,
                     writer_args=writer_args,
                     optimizer=optimizer, lr_scheduler=lr_scheduler,
                     model_ema=model_ema_module,
                     pre_conditioner=kfac_optimizer, writer=writer)
