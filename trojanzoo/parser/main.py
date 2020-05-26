# -*- coding: utf-8 -*-

from .parser import Parser

import torch

from trojanzoo.config import Config
env = Config.env


class Parser_Main(Parser):

    def __init__(self, *args, name='main'):
        super().__init__(*args, name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--device', dest='device')
        parser.add_argument('--cache_threshold', dest='cache_threshold',
                            type=float)

    @staticmethod
    def get_module(device: str = None, cache_threshold: float = None):
        env['num_gpus'] = 0
        if device in [None, 'gpu', 'cuda'] or 'cuda' in device:
            if torch.cuda.is_available():
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
                env['num_gpus'] = torch.cuda.device_count()
            elif device is not None:
                raise Exception('CUDA is not available on this device.')
        env['cache_threshold'] = Config.config['general']['cache_threshold'] if cache_threshold is None else cache_threshold
