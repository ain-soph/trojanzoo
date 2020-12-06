# -*- coding: utf-8 -*-

from .imc import Parser_IMC


class Parser_IMC_Adaptive(Parser_IMC):
    r"""IMC Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'imc'``
    """
    attack = 'imc_adaptive'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--abs_weight', dest='abs_weight', type=float)
        parser.add_argument('--strip_percent', dest='strip_percent', type=float)
