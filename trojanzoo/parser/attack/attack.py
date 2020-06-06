# -*- coding: utf-8 -*-

from ..parser import Parser
from trojanzoo.utils.loader import get_attack


class Parser_Attack(Parser):

    name = 'attack'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--iteration', dest='module_name', type=str)
        parser.add_argument('--early_stop', dest='early_stop',
                            action='store_true')
        parser.add_argument('--stop_confidence', dest='stop_confidence',
                            type=float)
        parser.add_argument('--output', dest='output', type=int)

    @staticmethod
    def get_module(**kwargs):
        return get_attack(**kwargs)