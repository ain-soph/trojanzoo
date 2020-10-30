# -*- coding: utf-8 -*-
from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Model_Agnostic(Parser_Defense_Backdoor):
    name: str = 'defense'
    defense = 'model_agnostic'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--size', dest='size', type=float,
                            help='size of the blocker')
        parser.add_argument('--N', dest='N', type=int,
                            help='iterations for random position generation')