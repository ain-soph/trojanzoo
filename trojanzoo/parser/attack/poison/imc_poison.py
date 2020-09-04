# -*- coding: utf-8 -*-

from .poison_basic import Parser_Poison_Basic


class Parser_IMC_Poison(Parser_Poison_Basic):
    r"""IMC Poison Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'imc_poison'``
    """
    attack = 'imc_poison'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--pgd_alpha', dest='pgd_alpha', type=float)
        parser.add_argument('--pgd_epsilon', dest='pgd_epsilon', type=float)
        parser.add_argument('--pgd_iteration', dest='pgd_iteration', type=int)
        parser.add_argument('--stop_conf', dest='stop_conf', type=float)

        parser.add_argument('--magnet', dest='magnet', action='store_true')
        parser.add_argument('--randomized_smooth', dest='randomized_smooth', action='store_true')
        parser.add_argument('--curvature', dest='curvature', action='store_true')
