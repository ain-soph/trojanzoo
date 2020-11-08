# -*- coding: utf-8 -*-
from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_NEO(Parser_Defense_Backdoor):
    name: str = 'defense'
    defense = 'neo'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--threshold_t', dest='threshold_t', type=float)
        parser.add_argument('--sample_num', dest='sample_num', type=int,
                            help='iterations for random position generation')
        parser.add_argument('--k_means_num', dest='k_means_num', type=int)
