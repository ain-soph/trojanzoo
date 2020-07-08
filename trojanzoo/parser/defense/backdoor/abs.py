# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_ABS(Parser_Defense_Backdoor):
    r"""ABS Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'abs'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--seed_num', dest='seed_num', type=int,
                            help='ABS seed number, defaults to config[abs][seed_num][dataset]=50.')
        parser.add_argument('--max_troj_size', dest='max_troj_size', type=int,
                            help='ABS max trojan trigger size (pixel number), defaults to config[abs][max_troj_size][dataset]=64.')
        parser.add_argument('--remask_epoch', dest='remask_epoch', type=int,
                            help='ABS optimizing epoch, defaults to config[abs][epoch][dataset]=1000.')
        parser.add_argument('--remask_lr', dest='remask_lr', type=float,
                            help='ABS optimization learning rate, defaults to config[abs][remask_lr][dataset]=0.1.')
        parser.add_argument('--remask_weight', dest='remask_weight', type=float,
                            help='ABS optimization remask loss weight, defaults to config[abs][remask_weight][dataset]=0.1.')
