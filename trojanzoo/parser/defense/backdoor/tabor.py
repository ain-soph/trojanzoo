# -*- coding: utf-8 -*-


from .neural_cleanse import Parser_Neural_Cleanse

class Parser_Tabor(Parser_Neural_Cleanse):
    r"""Tabor Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'tabor'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--hyperparams', dest='hyperparams', type=list,
                            help='the hyperparameters of  all regularization terms, defaults to config[tabor][hyperparams] = [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2].')
