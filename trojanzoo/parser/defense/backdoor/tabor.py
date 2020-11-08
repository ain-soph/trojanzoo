'''
Author: your name
Date: 2020-09-30 22:14:44
LastEditTime: 2020-09-30 22:16:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /xs/Trojan-Zoo/trojanzoo/parser/defense/backdoor/neural_cleanse.py
'''
# -*- coding: utf-8 -*-

from .neural_cleanse import Parser_Neural_Cleanse


class Parser_TABOR(Parser_Neural_Cleanse):
    r"""TABOR Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'tabor'
