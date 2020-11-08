# -*- coding: utf-8 -*-


from .badnet import Parser_BadNet


class Parser_Hidden_Trigger(Parser_BadNet):
    r"""Hidden Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'hidden_trigger'``
    """
    attack = 'hidden_trigger'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                            help='the chosen feature layer patched by trigger where distance to poisoned images is minimized, defaults to config[hidden][preprocess_layer]=features')

        parser.add_argument('--pgd_alpha', dest='pgd_alpha', type=float,
                            help='the learning rate to generate poison images, defaults to config[hidden][poison_lr]=0.01')
        parser.add_argument('--pgd_epsilon', dest='pgd_epsilon', type=int,
                            help='the perturbation threshold in input space, defaults to config[hidden][epsilon]=16')
        parser.add_argument('--pgd_iteration', dest='pgd_iteration', type=int,
                            help='the iteration number to generate one poison image, defaults to config[hidden][poison_generation_iteration]=5000')
