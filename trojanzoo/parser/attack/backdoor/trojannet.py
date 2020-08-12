# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_Trojan_Net(Parser_Attack):
    attack = 'trojannet'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--syn_backdoor_map', dest='syn_backdoor_map', nargs='+', type=int,
                            help='synthesize backdoor map')
        parser.add_argument('--model_save_path', dest='model_save_path', type=str, help='model_save_path')
        # parser.add_argument('--attack_class', dest='attack_class', type=int, help='attack_class', default=0)
