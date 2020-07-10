# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_DeepInspect(Parser_Defense_Backdoor):
    r"""DeepInspect Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'deepinspect'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--sample_num', dest='sample_num', type=int,
                            help='number of samples used to add trigger')

        parser.add_argument('--epoch', dest='epoch', type=int,
                            help='optimizing epoch')

        parser.add_argument('--lr', dest='lr', type=int,
                            help='optimizing learning rate')

        parser.add_argument('--gamma_1', dest='gamma_1', type=int,
                            help='control effect of GAN loss')

        parser.add_argument('--gamma_2', dest='gamma_2', type=int,
                            help='control effect of perturbation loss')