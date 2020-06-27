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
        parser.add_argument('--poison_num', dest='poison_num', type=int,
                            help='the number of poisoned images')
        parser.add_argument('--poison_iteration', dest='poison_iteration', type=int,
                            help='the iteration number to generate one poison image')
        parser.add_argument('--poison_lr', dest='poison_lr', type=float,
                            help='the learning rate to generate poison images')

        parser.add_argument('--lr_decay', dest='lr_decay', action='store_true',
                            help='the learning rate decays with iterations')
        parser.add_argument('--decay_iteration', dest='decay_iteration', type=int,
                            help='the iteration interval of lr decay')
        parser.add_argument('--decay_ratio', dest='decay_ratio', type=float,
                            help='the learning rate decay ratio')

        parser.add_argument('--ynt_ratio', dest='ynt_ratio', type=float,
                            help='the ratio of selected non-target images for trigger generation')
        parser.add_argument('--mark_area_ratio', dest='mark_area_ratio', type=float,
                            help='the ratio of trigger mark area to original image area')
        parser.add_argument('--val_ratio', dest='val_ratio', type=float,
                            help='the ratio of validate set from poisoned data, where 1-val_ratio will be ratio of trainset')
        parser.add_argument('--fine_tune_set_ratio', dest='fine_tune_set_ratio', type=float,
                            help='the ratio of dataset from clean images, used for fine-tuning')