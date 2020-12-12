# -*- coding: utf-8 -*-

from trojanzoo.utils.param import Param
from trojanzoo.utils.config import Config
from trojanzoo.utils.output import ansi, prints

import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np
import random
import argparse


class Env:
    env = Param()

    @classmethod
    def initialize(cls):
        num_gpus: int = torch.cuda.device_count()
        device: str = 'cuda' if num_gpus else 'cpu'
        cls.env.update(num_gpus=num_gpus, device=device)
        cls.env.update(**Config.config['env'])

    @classmethod
    def summary(cls, indent: int = 0):
        prints(cls.env, indent=indent)


def add_argument(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('{yellow}env{reset}'.format(**ansi))
    group.add_argument('--config', dest='config_path',
                       help='cmd config file path. (``package < project < cmd_config < cmd_param``)')
    group.add_argument('--data_dir', dest='data_dir',
                       help='data directory to contain datasets and models, defaults to config[env][data_dir]')
    group.add_argument('--result_dir', dest='result_dir',
                       help='result directory to save results, defaults to config[env][result_dir]')
    group.add_argument('--memory_dir', dest='memory_dir',
                       help='memory directory to contain datasets on tmpfs (optional), defaults to config[env][memory_dir]')

    group.add_argument('--seed', dest='seed', type=int,
                       help='the random seed for numpy, torch and cuda, defaults to config[env][seed]=1228')
    group.add_argument('--cache_threshold', dest='cache_threshold', type=float,
                       help='the threshold (MB) to call torch.cuda.empty_cache(), defaults to config[env][cache_threshold]=None (never).')

    group.add_argument('--device', dest='device',
                       help='set to \'cpu\' to force cpu-only and \'gpu\', \'cuda\' for gpu-only, defaults to \'auto\'.')
    group.add_argument('--benchmark', dest='benchmark', action='store_true',
                       help='use torch.backends.cudnn.benchmark to accelerate without deterministic, defaults to False.')
    group.add_argument('--verbose', dest='verbose', action='store_true',
                       help='show arguments and module information, defaults to False.')
    group.add_argument('--color', dest='color', action='store_true',
                       help='Colorful Output, defaults to False.')
    group.add_argument('--tqdm', dest='tqdm', action='store_true',
                       help='Show tqdm Progress Bar, defaults to False.')
    return group


def create(config_path: str = None, seed: int = None, device: str = None, benchmark: bool = None, **kwargs):
    Config.set_config_path('cmd', config_path)
    Config.update()
    Env.env.update(config_path=config_path, **Config.config['env'])
    if seed is None and 'seed' in Env.env.keys():
        seed = Env.env['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if isinstance(device, str):
        if device == 'cpu':
            Env.env.update(device=device, num_gpus=0)
        if 'cuda:' in device:
            Env.env.update(device=device, num_gpus=1)
        if device == 'auto':
            device = None
    if benchmark is None and 'benchmark' in Env.env.keys():
        benchmark = Env.env['benchmark']
    if benchmark:
        torch.backends.cudnn.benchmark = benchmark
    Env.env.update(seed=seed, device=device, benchmark=benchmark, **kwargs)


Env.initialize()
env = Env.env
