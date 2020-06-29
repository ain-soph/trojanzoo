# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser


class Parser_AdvMind(Parser):
    r"""AdvMind Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific attack name (lower-case).
    """
    name = 'defense'
    defense = None

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--attack_adapt', dest='attack_adapt', action='store_true',
                            help='Adaptive attack to add fake queries.')
        parser.add_argument('--fake_percent', dest='fake_percent', type=float,
                            help='fake query percentage.')
        parser.add_argument('--dist', dest='dist', type=float,
                            help='fake query noise std.')
        parser.add_argument('--defend_adapt', dest='defend_adapt', action='store_true',
                            help='Robust location M-estimator.')
        parser.add_argument('--active', dest='active', action='store_true',
                            help='Proactive solicitation.')
        parser.add_argument('--active_percent', dest='active_percent', type=float,
                            help='Active gradient weight.')
