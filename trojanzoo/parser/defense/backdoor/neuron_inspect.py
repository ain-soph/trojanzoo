# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Neuron_Inspect(Parser_Defense_Backdoor):
    r"""Deep Inspect Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'neuron_inspect'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--lambd_sp', dest='lambd_sp', type=float,
                            help='control sparse feature')
        parser.add_argument('--lambd_sm', dest='lambd_sm', type=float,
                            help='control smooth feature')
        parser.add_argument('--lambd_pe', dest='lambd_pe', type=float,
                            help='control persistence feature')

        parser.add_argument('--thre', dest='thre', type=float,
                            help='Threshold for calculating persistence feature')

        parser.add_argument('--sample_ratio', dest='sample_ratio', type=float,
                            help='sample ratio from the full training data')

        