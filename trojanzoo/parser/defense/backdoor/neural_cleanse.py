'''
Author: your name
Date: 2020-09-30 22:14:44
LastEditTime: 2020-09-30 22:16:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /xs/Trojan-Zoo/trojanzoo/parser/defense/backdoor/neural_cleanse.py
'''
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
        parser.add_argument('--penalize', dest='penalize', type=bool,
                            help='add the regularization terms, nc to tabor, defaults to config[neural_cleanse][penalize]=False.')
        parser.add_argument('--hyperparams', dest='hyperparams', type=list,
                            help='the hyperparameters of  all regularization terms, defaults to config[tabor][hyperparams] = [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2].')
