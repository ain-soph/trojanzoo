# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_Trojan_Net(Parser_Attack):
    attack = 'trojannet'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--syn_backdoor_map', dest='syn_backdoor_map', type=tuple,
                            help='synthesize backdoor map')