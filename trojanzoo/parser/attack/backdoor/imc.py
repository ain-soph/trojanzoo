# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_IMC(Parser_BadNet):
    r"""IMC Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'imc'``
    """
    attack = 'imc'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--pgd_alpha', dest='pgd_alpha', type=float)
        parser.add_argument('--pgd_epsilon', dest='pgd_epsilon', type=float)
        parser.add_argument('--pgd_iteration', dest='pgd_iteration', type=int)
