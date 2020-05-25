# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.output import prints


class Parser_Dataset(Parser):

    def __init__(self, name='dataset'):
        super().__init__(name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-d', '--dataset', dest='module_name',
                            default='cifar10')
        parser.add_argument('--batch_size', dest='batch_size',
                            type=int)

    def parse_args(self, *args, **kwargs):
        return super().parse_args(*args, **kwargs)

    def set_module(self, **kwargs):
        super().set_module(output=self.output, **kwargs)
