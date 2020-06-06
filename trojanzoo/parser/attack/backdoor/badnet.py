# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_BadNet(Parser_Attack):

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--target_class', dest='target_class', type=int)
        parser.add_argument('--percent', dest='percent', type=float)
