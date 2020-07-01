# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_PGD(Parser_Attack):
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
        parser.add_argument('--epsilon', dest='epsilon', type=float,
                            help='Projection norm constraint, defaults to config[pgd][epsilon][dataset]=8.0/255')
        parser.add_argument('--iteration', dest='iteration', type=int,
                            help='Attack Iteration, defaults to config[pgd][iteration][dataset]=20')
        parser.add_argument('--stop_threshold', dest='stop_threshold', type=float,
                            help='early stop confidence, defaults to config[pgd][stop_threshold][dataset]=None')
        parser.add_argument('--target_idx', dest='target_idx', type=int,
                            help='Target label order in original classification, defaults to config[pgd][target_idx][dataset]=1 ' +
                                 '(0 for untargeted attack, 1 for most possible class, -1 for most unpossible class)')

        parser.add_argument('--grad_method', dest='grad_method',
                            help='gradient estimation method, defaults to config[pgd][grad_method][dataset]=\'white\'')
        parser.add_argument('--query_num', dest='query_num', type=int,
                            help='query numbers for black box gradient estimation, defaults to config[pgd][query_num][dataset]=100.')
        parser.add_argument('--sigma', dest='sigma', type=float,
                            help='gaussian sampling std for black box gradient estimation, defaults to config[pgd][sigma][dataset]=1e-3')
