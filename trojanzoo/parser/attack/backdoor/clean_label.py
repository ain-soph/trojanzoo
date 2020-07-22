# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_Clean_Label(Parser_Attack):
    r"""Clean Label Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'clean_label'``
    """
    attack = 'clean_label'
    
    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                            help='the chosen layer used to generate adversarial example, defaults to config[clean_label][preprocess_layer]=classifier')
        parser.add_argument('--poison_generation_method', dest='poison_generation_method', type=str,
                            help='the chosen method to generate poisoned sample, defaults to config[clean_label][poison_generation_method]=PGD')

        parser.add_argument('--tau', dest='tau', type=float,
                            help='the interpolation constant used to balance source imgs and target imgs, defaults to config[clean_label][tau]=0.4')
        parser.add_argument('--epsilon', dest='epsilon', type=float,
                            help='the perturbation bound in input space, defaults to config[clean_label][epsilon]=0.1, 300/(3*32*32)')

        parser.add_argument('--target_class', dest='target_class', type=int,
                            help='the class of poisoned data, defaults to config[clean_label][target_class]=9')

        parser.add_argument('--poison_ratio', dest='poison_ratio', type=float,
                            help='the ratio of source class sample used to generate poisoned sample, only for source class data,not whole training data, defaults to config[clean_label][poison_ratio]=0.001')
        parser.add_argument('--noise_dim', dest='noise_dim', type=int,
                            help='the dimension of the input in the generator, defaults to config[clean_label][noise_dim]=100')

        parser.add_argument('--generator_iters', dest='generator_iters',type=int,help=' the epoch for training the generator, defaults to config[clean_label][generator_iters]=1000')
        parser.add_argument('--critic_iter', dest='critic_iter',type=int,help=' the critic iterations per generator training iteration, defaults to config[clean_label][critic_iter]=5')
