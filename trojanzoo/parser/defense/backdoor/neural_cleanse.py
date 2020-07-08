# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Neural_Cleanse(Parser_Defense_Backdoor):
    r"""Neural Cleanse Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'neural_cleanse'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--nc_epoch', dest='epoch', type=int,
                            help='neural cleanse optimizing epoch, defaults to config[neural_cleanse][epoch][dataset]=10.')
