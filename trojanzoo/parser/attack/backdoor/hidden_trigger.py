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
        parser.add_argument('--epsilon', dest='epsilon', type=int,
                            help='the perturbation threshold in input space, defaults to config[hidden][epsilon]=16')

        parser.add_argument('--poison_num', dest='poison_num', type=int,
                            help='the number of poisoned images, defaults to config[hidden][poisoned_image_num]=100')
        parser.add_argument('--poison_iteration', dest='poison_iteration', type=int,
                            help='the iteration number to generate one poison image, defaults to config[hidden][poison_generation_iteration]=5000')
        parser.add_argument('--poison_lr', dest='poison_lr', type=float,
                            help='the learning rate to generate poison images, defaults to config[hidden][poison_lr]=0.01')

        parser.add_argument('--lr_decay', dest='lr_decay', action='store_true',
                            help='the learning rate decays with iterations, defaults to config[hidden][decay]=True')
        parser.add_argument('--decay_iteration', dest='decay_iteration', type=int,
                            help='the iteration interval of lr decay, defaults to config[hidden][decay_iteration]=2000')
        parser.add_argument('--decay_ratio', dest='decay_ratio', type=float,
                            help='the learning rate decay ratio, defaults to config[hidden][decay_ratio]=0.95')
