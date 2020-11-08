# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_Term_Study(Parser_BadNet):
    r"""Term Study Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'term_study'``
    """
    attack = 'term_study'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--term', dest='term', type=str)

        parser.add_argument('--pgd_alpha', dest='pgd_alpha', type=float)
        parser.add_argument('--pgd_epsilon', dest='pgd_epsilon', type=float)
        parser.add_argument('--pgd_iteration', dest='pgd_iteration', type=int)

        parser.add_argument('--class_sample_num', dest='class_sample_num', type=int,
                            help='the number of sampled images per class, defaults to config[latent_backdoor][class_sample_num][dataset]=100')
        parser.add_argument('--mse_weight', dest='mse_weight', type=float,
                            help='the weight of mse loss during retraining, defaults to config[latent_backdoor][mse_weight][dataset]=100')
        parser.add_argument('--preprocess_layer', dest='preprocess_layer',
                            help='the chosen feature layer patched by trigger, defaults to \'features\'')
        parser.add_argument('--preprocess_epoch', dest='preprocess_epoch', type=int,
                            help='preprocess optimization epoch')
        parser.add_argument('--preprocess_lr', dest='preprocess_lr', type=float,
                            help='preprocess learning rate')
