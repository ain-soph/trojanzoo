# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.utils.param import Param

import random
import numpy as np
import torch

from trojanzoo.utils import Config
env = Config.env


class Parser_Config(Parser):
    r"""Config Parser to update ``config`` and ``env`` according to cmd parameters.

    Attributes:
        name (str): ``'config'``
    """
    name = 'config'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--config', dest='config',
                            help='cmd config file path. (``package < project < cmd_config < cmd_param``)')

        parser.add_argument('--data_dir', dest='data_dir',
                            help='data directory to contain datasets and models, defaults to config[env][data_dir]')
        parser.add_argument('--result_dir', dest='result_dir',
                            help='result directory to save results, defaults to config[env][result_dir]')
        parser.add_argument('--memory_dir', dest='memory_dir',
                            help='memory directory to contain datasets on tmpfs (optional), defaults to config[env][memory_dir]')

        parser.add_argument('--seed', dest='seed', type=int,
                            help='the random seed for numpy, torch and cuda, defaults to config[env][seed]=1228')
        parser.add_argument('--cache_threshold', dest='cache_threshold', type=float,
                            help='the threshold (MB) to call torch.cuda.empty_cache(), defaults to config[env][cache_threshold]=None (never).')

    @staticmethod
    def get_module(config: str = None, **kwargs) -> Param:
        # type: (str, dict) -> Param  # noqa
        r"""
        update ``config`` according to ``cmd_config`` (``--config``).

        update ``env`` according to listed ``cmd_param`` (e.g. ``--data_dir``).

        Args:
            config (str):  cmd config file path. Default: None.
        Returns:
            :class:`Param`
        """
        Config.update(cmd_path=config)
        Config.update_env(**kwargs)
        random.seed(env['seed'])
        np.random.seed(env['seed'])
        torch.manual_seed(env['seed'])
        torch.cuda.manual_seed_all(env['seed'])
        return Config.config
