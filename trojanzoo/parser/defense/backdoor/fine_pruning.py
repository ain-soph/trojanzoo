# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Fine_Pruning(Parser_Defense_Backdoor):
    r"""Fine Pruning Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'fine_pruning'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--clean_image_num', dest='clean_image_num', type=int,
                            help=' the number of sampled clean image to prune and finetune the model, defaults to config[Fine_Pruning][clean_image_num]=50')
        parser.add_argument('--prune_ratio', dest='prune_ratio', type=float,
                            help=' the ratio of neurons to prune, defaults to config[Fine_Pruning][prune_ratio ]=0.02')
