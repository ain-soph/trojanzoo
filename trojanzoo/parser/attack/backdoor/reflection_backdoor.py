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
        parser.add_argument('--candidate_num', dest='candidate_num', type=int,
                            help='number of candidate images')
        parser.add_argument('--selection_num', dest='selection_num', type=int,
                            help='number of adv images')
        parser.add_argument('--selection_iter', dest='selection_iter', type=int,
                            help='selection iteration to find optimal reflection images as trigger')
        parser.add_argument('--inner_epoch', dest='inner_epoch', type=int,
                            help='retraining epoch during trigger selection')
