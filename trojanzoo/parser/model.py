# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.loader import get_model
from trojanzoo.model import Model


class Parser_Model(Parser):

    def __init__(self, name='model'):
        super().__init__(name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model', dest='module_name', type=str)
        parser.add_argument('--layer', dest='layer', type=int)
        parser.add_argument('--pretrain', dest='pretrain',
                            action='store_true')
        parser.add_argument('--adv_train', dest='adv_train',
                            action='store_true')

    def get_module(self, **kwargs) -> Model:
        return get_model(**kwargs)
