# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_STRIP(Parser_Defense_Backdoor):
    r"""Neural Cleanse Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'strip'
