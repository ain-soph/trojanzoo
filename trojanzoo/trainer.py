#!/usr/bin/env python3

from trojanzoo.models import Model
from trojanzoo.configs import config
from trojanzoo.environ import env
from trojanzoo.utils.model import ExponentialMovingAverage
from trojanzoo.utils.module import get_name, BasicObject
from trojanzoo.utils.output import ansi, prints
from trojanzoo.utils.fim import KFAC, EKFAC

from typing import TYPE_CHECKING
from typing import Any
from trojanzoo.configs import Config    # TODO: python 3.10
from trojanzoo.datasets import Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import argparse
if TYPE_CHECKING:
    pass


class Trainer(BasicObject):
    r"""A dict-like class to contain training arguments
    which supports attribute-like view as well.

    It inherits :class:`trojanzoo.utils.module.BasicObject`.

    Note:
        The most common usage is ``train(**trainer)``.
        See :meth:`keys()` for details.

    Attributes:
        optim_args (dict[str, Any]): optimizer arguments.
        train_args (dict[str, Any]): train function arguments.
        writer_args (dict[str, Any]):
            :any:`torch.utils.tensorboard.writer.SummaryWriter` arguments.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        lr_scheduler (``torch.optim.lr_scheduler._LRScheduler`` | None):
            LR_Scheduler instance.
        model_ema (~trojanzoo.utils.model.ExponentialMovingAverage | None):
            Exponential Moving Average instance.
        pre_conditioner (~trojanzoo.utils.fim.KFAC | ~trojanzoo.utils.fim.EKFAC | None):
            Pre-conditioner instance.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | None):
            Tensorboard summary writer instance.
    """
    name = 'trainer'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        r"""Add trainer arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
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

        group.add_argument('--pre_conditioner', choices=[None, 'kfac', 'ekfac'],
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
                 pre_conditioner: None | KFAC | EKFAC = None,
                 writer=None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['trainer'] = ['optim_args', 'train_args', 'writer_args',
                                      'optimizer', 'lr_scheduler',
                                      'pre_conditioner', 'model_ema',
                                      'writer']
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

    def keys(self) -> list[str]:
        r"""Keys include:

            * | All attributes exclude
              | ``['optim_args', 'train_args', 'writer_args']``
            * train_args

        Returns:
            list[str]: The list of keys.
        """
        keys = self.param_list['trainer'].copy()
        keys.remove('optim_args')
        keys.remove('train_args')
        keys.remove('writer_args')
        keys.extend(self.train_args.keys())
        return keys

    def summary(self, indent: int = None):
        indent = indent if indent is not None else self.indent
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(
            self.name, **ansi), indent=indent)
        prints('{yellow}{0}{reset}'.format(self.__class__.__name__, **ansi), indent=indent)
        for key in self.param_list['trainer']:
            value = getattr(self, key)
            if value:
                prints('{green}{0:<10s}{reset}'.format(key, **ansi),
                       indent=indent + 10)
                if isinstance(value, dict):
                    value = {k: str(v).split('\n')[0] for k, v in value.items()}
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


def add_argument(parser: argparse.ArgumentParser,
                 ClassType: type[Trainer] = Trainer
                 ) -> argparse._ArgumentGroup:
    r"""
    | Add trainer arguments to argument parser.
    | For specific arguments implementation, see :meth:`Trainer.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        ClassType (type[Trainer]): The trainer type.
            Defaults to :class:`Trainer`.

    Returns:
        argparse._ArgumentGroup: The argument group.
    """
    group = parser.add_argument_group('{yellow}trainer{reset}'.format(**ansi))
    return ClassType.add_argument(group)


def create(dataset_name: None | str = None,
           dataset: None | str | Dataset = None,
           model: None | Model = None,
           model_ema: None | bool = False,
           pre_conditioner: None | str = None,
           tensorboard: None | bool = None,
           ClassType: type[Trainer] = Trainer,
           config: Config = config, **kwargs):
    r"""
    | Create a trainer instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | For trainer implementation, see :class:`Trainer`.

    Args:
        dataset_name (str): The dataset name.
        dataset (str | trojanzoo.datasets.Dataset):
            Dataset instance
            (required for :attr:`model_ema`)
            or dataset name
            (as the alias of `dataset_name`).
        model (trojanzoo.models.Model): Model instance.
        model_ema (bool): Whether to use
            :class:`~trojanzoo.utils.model.ExponentialMovingAverage`.
            Defaults to ``False``.
        pre_conditioner (str): Choose from

            * ``None``
            * ``'kfac'``: :class:`~trojanzoo.utils.fim.KFAC`
            * ``'ekfac'``: :class:`~trojanzoo.utils.fim.EKFAC`

            Defaults to ``None``.
        tensorboard (bool): Whether to use
            :any:`torch.utils.tensorboard.writer.SummaryWriter`.
            Defaults to ``False``.
        ClassType (type[Trainer]): The trainer class type.
            Defaults to :class:`Trainer`.
        config (Config): The default parameter config.
        **kwargs: The keyword arguments in keys of
            ``['optim_args', 'train_args', 'writer_args']``.

    Returns:
        Trainer: The trainer instance.
    """
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
    match optim_args['parameters']:
        case 'features':
            module = module.features
        case 'classifier':
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

    match pre_conditioner:
        case 'kfac':
            kfac_optimizer = KFAC(module)
        case 'ekfac':
            kfac_optimizer = EKFAC(module)
        case _:
            kfac_optimizer = None

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
