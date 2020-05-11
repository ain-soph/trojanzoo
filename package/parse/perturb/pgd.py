# -*- coding: utf-8 -*-

from package.utils.utils import Module, Param
from . import Parser_Perturb

param = Param(default=Module(alpha=3.0/255, epsilon=8.0/255, iteration=20))


class Parser_PGD(Parser_Perturb):

    def __init__(self, *args, param=param, **kwargs):
        super().__init__(*args, param=param, **kwargs)

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.set_defaults(module_name='pgd')
        parser.add_argument('--alpha', dest='alpha',
                            default=None, type=float)
        parser.add_argument('--epsilon', dest='epsilon',
                            default=None, type=float)
        parser.add_argument('--targeted', dest='targeted',
                            default=True, type=bool)
        parser.add_argument('--mode', dest='mode',
                            default='white')
