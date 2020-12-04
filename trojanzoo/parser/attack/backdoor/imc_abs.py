# -*- coding: utf-8 -*-

from .imc import Parser_IMC


class Parser_IMC_ABS(Parser_IMC):
    r"""IMC Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'imc'``
    """
    attack = 'imc_abs'
