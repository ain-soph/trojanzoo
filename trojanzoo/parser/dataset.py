# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.loader import get_dataset
from trojanzoo.dataset import Dataset


class Parser_Dataset(Parser):

    def __init__(self, name='dataset'):
        super().__init__(name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-d', '--dataset', dest='module_name', type=str)
        parser.add_argument('--batch_size', dest='batch_size', type=int)
        parser.add_argument('--download', dest='download',
                            action='store_true', default=False)

    def get_module(self, **kwargs) -> Dataset:
        return get_dataset(**kwargs)
