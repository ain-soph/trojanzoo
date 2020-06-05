# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.loader import get_mark


class Parser_Mark(Parser):

    name = 'mark'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--edge_color', dest='edge_color')
        parser.add_argument('--mark_path', dest='mark_path')

        parser.add_argument('--mark_alpha', dest='mark_alpha', type=float)
        parser.add_argument('--height', dest='height', type=int)
        parser.add_argument('--width', dest='width', type=int)
        parser.add_argument('--height_ratio', dest='height_ratio', type=float)
        parser.add_argument('--width_ratio', dest='width_ratio', type=float)
        parser.add_argument('--height_offset', dest='height_offset', type=int)
        parser.add_argument('--width_offset', dest='width_offset', type=int)

    @staticmethod
    def get_module(**kwargs):
        return get_mark(**kwargs)
