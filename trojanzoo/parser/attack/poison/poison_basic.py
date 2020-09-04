# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_Poison_Basic(Parser_Attack):
    r"""Poison Baisc Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'poison_basic'``
    """
    attack = 'poison_basic'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--percent', dest='percent', type=float,
                            help='malicious training data injection probability for each batch, defaults to config[badnet][target_class][dataset]=0.1')
        parser.add_argument('--target_idx', dest='target_idx', type=int,
                            help='Target label order in original classification, defaults to config[pgd][target_idx][dataset]=1 '
                                 '(0 for untargeted attack, 1 for most possible class, -1 for most unpossible class)')
