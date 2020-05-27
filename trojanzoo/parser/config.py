# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.config import Config


class Parser_Config(Parser):

    def __init__(self, *args, name='config'):
        super().__init__(*args, name=name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--config', dest='config')

    def get_module(self, config=None):
        Config.update(cmd_path=config)
        return Config.config
