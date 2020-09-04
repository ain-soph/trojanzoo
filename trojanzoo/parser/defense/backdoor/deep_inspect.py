# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Deep_Inspect(Parser_Defense_Backdoor):
    r"""Deep Inspect Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'deep_inspect'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--sample_ratio', dest='sample_ratio', type=float,
                            help='sample ratio from the full training data')

        parser.add_argument('--noise_dim', dest='noise_dim', type=int,
                            help='GAN noise dimension')

        parser.add_argument('--remask_epoch', dest='remask_epoch', type=int,
                            help='Remask optimizing epoch')
        parser.add_argument('--remask_lr', dest='remask_lr', type=float,
                            help='Remask optimizing learning rate')

        parser.add_argument('--gamma_1', dest='gamma_1', type=float,
                            help='control effect of GAN loss')
        parser.add_argument('--gamma_2', dest='gamma_2', type=float,
                            help='control effect of perturbation loss')
