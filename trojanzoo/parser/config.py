# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.config import Config
env = Config.env


class Parser_Config(Parser):

    name = 'config'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--config', dest='config')

        parser.add_argument('--data_dir', dest='data_dir')
        parser.add_argument('--result_dir', dest='result_dir')
        parser.add_argument('--memory_dir', dest='memory_dir')

        parser.add_argument('--seed', dest='seed', type=int)
        parser.add_argument('--cache_threshold', dest='cache_threshold',
                            type=float)

    @staticmethod
    def get_module(config=None, **kwargs):
        Config.update(cmd_path=config)
        Config.update_env(**kwargs)
        return Config.config
