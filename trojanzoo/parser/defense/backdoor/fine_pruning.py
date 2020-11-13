# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Fine_Pruning(Parser_Defense_Backdoor):
    r"""Fine Pruning Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'fine_pruning'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--prune_ratio', dest='prune_ratio', type=float,
                            help='the ratio of neuron number to prune, defaults to config[fine_pruning][prune_ratio]=0.95')
