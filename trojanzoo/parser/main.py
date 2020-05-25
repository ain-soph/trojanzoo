# -*- coding: utf-8 -*-

from .parser import Parser

import torch


class Parser_Main(Parser):

    def __init__(self, *args, name='main'):
        super().__init__(*args, name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--device', dest='device')

    @staticmethod
    def get_module(device=None):
        if device in [None, 'gpu', 'cuda']:
            if torch.cuda.is_available():
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            elif device is not None:
                raise Exception('CUDA is not available on this device.')
