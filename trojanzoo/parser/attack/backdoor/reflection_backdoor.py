# -*- coding: utf-8 -*-


from .badnet import Parser_BadNet


class Parser_Reflection_Backdoor(Parser_BadNet):
    r"""Reflection Backdoor Attack Parser
    Attributes:
        name (str): ``'attack'``
        attack (str): ``'reflection_backdoor'``
    """
    attack = 'reflection_backdoor'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--reflect_num', dest='reflect_num', type=int,
                            help='number of reflection images')
        parser.add_argument('--selection_step', dest='selection_step', type=int,
                            help='number of selection step to find optimal reflection images as trigger')
        parser.add_argument('--poison_num', dest='poison_num',
                            help='number of posioned images in training/validation set')

