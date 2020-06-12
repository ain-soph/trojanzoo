# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.param import Module
from trojanzoo.model import Model

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from trojanzoo.utils import Config
config = Config.config


class Parser_Train(Parser):
    r"""Train Parser to get :class:`Optimizer`, :class:`_LRScheduler` and arguments for training.

    Attributes:
        name (str): ``'train'``
    """
    name = 'train'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--epoch', dest='epoch', type=int,
                            help='training epochs, defaults to config[train][epoch][dataset].')
        parser.add_argument('--lr', dest='lr', type=float,
                            help='learning rate, defaults to config[train][lr][dataset]=0.1.')
        parser.add_argument('--parameters', dest='parameters', default='full',
                            help='training parameters (\'features\', \'classifier\', \'full\'), defaults to config[train][parameters][dataset]\'full\'.')
        parser.add_argument('--optim_type', dest='optim_type',
                            help='optimizer type, defaults to config[train][optim_type][dataset]=SGD.')
        parser.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true',
                            help='use torch.optim.lr_scheduler.StepLR.')
        parser.add_argument('--step_size', dest='step_size', type=int,
                            help='step_size passed to torch.optim.lr_scheduler.StepLR, defaults to config[train][step_size][dataset]=50.')
        parser.add_argument('--validate_interval', dest='validate_interval', type=int,
                            help='validate interval during training epochs, defaults to config[train][validate_interval][dataset]=10.')
        parser.add_argument('--save', dest='save', action='store_true',
                            help='save training results.')

    @staticmethod
    def get_module(model: Model, **kwargs) -> (Optimizer, _LRScheduler, dict):
        # type: (Model, dict) -> (Optimizer, _LRScheduler, dict)  # noqa
        r"""
        get :class:`Optimizer`, :class:`_LRScheduler` and arguments for training

        by splitting ``kwargs`` to :meth:`Model.define_optimizer` and :func:`Model._train`.

        Args:
            model (Model):

        Returns:
            (:class:`Optimizer`, :class:`_LRScheduler`, training arguments)
        """

        dataset = 'default'
        if 'dataset' in kwargs.keys():
            dataset = kwargs['dataset']
            if not isinstance(dataset, str):
                dataset = dataset.name
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
