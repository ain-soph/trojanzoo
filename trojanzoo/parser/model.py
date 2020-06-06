# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.loader import get_model


class Parser_Model(Parser):

    name='model'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-m', '--model', dest='module_name', type=str)
        parser.add_argument('--layer', dest='layer', type=int)
        parser.add_argument('--pretrain', dest='pretrain', action='store_true')
        parser.add_argument('--official', dest='official', action='store_true')
        parser.add_argument('--adv_train', dest='adv_train',
                            action='store_true')

    @staticmethod
    def get_module(**kwargs):
        return get_model(**kwargs)
