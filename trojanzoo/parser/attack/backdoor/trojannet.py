# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_TrojanNet(Parser_BadNet):
    attack = 'trojannet'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--select_point', dest='select_point', type=int,
                            help='the number of select_point, defaults to config[trojannet][select_point]=2')