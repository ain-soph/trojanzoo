# -*- coding: utf-8 -*-

from package.utils.utils import Module, Param
from . import Parser_Perturb

param = Param(default=Module(lr=0.001, epoch=1, poison_percent=0.01))


class Parser_Poison(Parser_Perturb):

    def __init__(self, *args, param=param, **kwargs):
        super().__init__(*args, param=param, **kwargs)

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.set_defaults(module_name='poison')
        parser.add_argument('--lr', dest='lr',
                            default=None, type=float)
        parser.add_argument('--epoch', dest='epoch',
                            default=None, type=int)
        parser.add_argument('--poison_percent', dest='poison_percent',
                            default=None, type=float)
        parser.add_argument('--poison_num', dest='poison_num',
                            default=None, type=float)
        parser.add_argument('--train_opt', dest='train_opt',
                            default='partial')
        parser.add_argument('--optim_type', dest='optim_type',
                            default='Adam')
        parser.add_argument('--lr_scheduler', dest='lr_scheduler',
                            default=False, action='store_true')
        parser.add_argument('--early_stop', dest='early_stop',
                            default=False, action='store_true')
