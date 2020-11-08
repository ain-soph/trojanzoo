# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Adv_Train(Parser_Defense_Backdoor):
    r"""AdvMind Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'adv_train'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--pgd_alpha', dest='pgd_alpha', type=float)
        parser.add_argument('--pgd_epsilon', dest='pgd_epsilon', type=float)
        parser.add_argument('--pgd_iteration', dest='pgd_iteration', type=int)
