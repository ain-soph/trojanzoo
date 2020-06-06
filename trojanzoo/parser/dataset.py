# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.loader import get_dataset


class Parser_Dataset(Parser):

    name='dataset'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-d', '--dataset', dest='module_name', type=str)
        parser.add_argument('--batch_size', dest='batch_size', type=int)
        parser.add_argument('--num_workers', dest='num_workers', type=int)
        parser.add_argument('--download', dest='download', action='store_true')

    @staticmethod
    def get_module(**kwargs):
        return get_dataset(**kwargs)
