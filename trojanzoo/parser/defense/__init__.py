# -*- coding: utf-8 -*-

from .adv import *
from .backdoor import *
from .. import Parser

from trojanzoo.defense import class_dict
import sys


class Parser_Defense(Parser):
    r"""Universal Defense Parser

    Attributes:
        name (str): ``'defense'``
    """
    name = 'defense'

    def __init__(self):
        argv = sys.argv
        try:
            idx = argv.index('--defense')
            self.defense = argv[idx + 1]
        except ValueError as e:
            print("You need to set '--defense' to call 'Parser_defense'. ")
            raise e

        pkg = __import__('trojanzoo.parser.defense', fromlist=['class_dict'])
        class_name: str = 'Parser_' + class_dict[self.defense]
        _class = getattr(pkg, class_name)
        self.parser: Parser = _class()

    def parse_args(self, args=None, namespace=None, **kwargs):
        return self.parser.parse_args(args=args, namespace=namespace, **kwargs)

    def get_module(self, **kwargs):
        return self.parser.get_module(**kwargs)
