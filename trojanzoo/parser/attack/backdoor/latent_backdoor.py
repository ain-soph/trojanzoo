# -*- coding: utf-8 -*-


from .badnet import Parser_BadNet


class Parser_Latent_Backdoor(Parser_BadNet):
    r"""Latent Backdoor Attack Parser
    Attributes:
        name (str): ``'attack'``
        attack (str): ``'latent_backdoor'``
    """
    attack = 'latent_backdoor'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
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
