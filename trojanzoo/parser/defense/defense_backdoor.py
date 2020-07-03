# -*- coding: utf-8 -*-

from .defense import Parser_Defense


class Parser_Defense_Backdoor(Parser_Defense):
    r"""Backdoor Defense Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name = 'defense'
    defense = 'defense_backdoor'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--original', dest='original', action='store_true',
                            help='load original clean model, defaults to False.')
