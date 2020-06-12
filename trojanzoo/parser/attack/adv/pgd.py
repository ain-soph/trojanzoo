# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_BadNet(Parser_Attack):
    r"""PGD Adversarial Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'pgd'``
    """
    attack = 'pgd'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--alpha', dest='alpha', type=float,
                            help='PGD learning rate per step, defaults to config[pgd][alpha][dataset]=3.0/255')
        parser.add_argument('--alpha', dest='alpha', type=float,
                            help='Projection norm constraint, defaults to config[pgd][epsilon][dataset]=8.0/255')
