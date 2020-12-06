# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_Unlearn(Parser_BadNet):
    r"""

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'unlearn'``
    """
    attack = 'unlearn'
