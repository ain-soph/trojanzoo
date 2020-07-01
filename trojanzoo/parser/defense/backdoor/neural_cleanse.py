# -*- coding: utf-8 -*-

from ..defense import Parser_Defense
from trojanzoo.dataset import Dataset
from trojanzoo.defense import Neural_Cleanse

from trojanzoo.utils import Config
config = Config.config


class Parser_Neural_Cleanse(Parser_Defense):
    r"""AdvMind Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name = 'defense'
    defense = 'neural_cleanse'

    @staticmethod
    def add_argument(parser):
        pass
