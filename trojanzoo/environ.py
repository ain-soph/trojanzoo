#!/usr/bin/env python3

from trojanzoo.configs import config
from trojanzoo.utils.module.param import Param
from trojanzoo.utils.module import get_name
from trojanzoo.utils.output import ansi

import torch
import numpy as np
import random

from typing import TYPE_CHECKING
from typing import Union    # TODO: python 3.10
from trojanzoo.configs import Config
import argparse
if TYPE_CHECKING:
    import torch.backends.cudnn


class Env(Param):

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        group.add_argument('--config', dest='config_path',
                           help='cmd config file path '
                           '(package < project < cmd_config < cmd_param)')

        group.add_argument('--seed', type=int,
                           help='the random seed for numpy, torch and cuda '
                           '(default: config[env][seed]=1228)')
        group.add_argument('--data_seed', type=int,
                           help='seed to process data')
        group.add_argument('--cache_threshold', type=float,
                           help='the threshold (MB) to call '
                           'torch.cuda.empty_cache(). None means never.'
                           '(default: config[env][cache_threshold]=None).')

        group.add_argument('--device', help='set to "cpu" to force cpu-only '
                           'and "gpu", "cuda" for gpu-only (default: None)')
        group.add_argument('--benchmark', action='store_true',
                           help='use torch.backends.cudnn.benchmark '
                           'to accelerate without deterministic')
        group.add_argument('--verbose', type=int, default=0,
                           help='show arguments and module information '
                           '(default: 0)')
        group.add_argument('--color', action='store_true',
                           help='show colorful output')
        group.add_argument('--tqdm', action='store_true',
                           help='show tqdm progress bar')
        return group


env = Env(default=None)


def add_argument(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('{yellow}env{reset}'.format(**ansi))
    env.add_argument(group)
    return group


def create(config_path: str = None, dataset_name: str = None,
           dataset: str = None,
           seed: int = None, data_seed: int = None, benchmark: bool = None,
           config: Config = config,
           cache_threshold: float = None, verbose: int = None,
           color: bool = None, tqdm: bool = None, **kwargs) -> Env:
    other_kwargs = {'data_seed': data_seed, 'cache_threshold': cache_threshold,
                    'verbose': verbose, 'color': color, 'tqdm': tqdm}
    config.update_cmd(config_path)
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None \
        else config.full_config['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)[
        'env'].update(other_kwargs)
    env.update(config_path=config_path, **result)
    ansi.switch(env['color'])
    if seed is None and 'seed' in env.keys():
        seed = env['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_gpus: int = torch.cuda.device_count()
    device: Union[str, int] = result['device']
    if device == 'none':
        device = None
    else:
        if device is None or device == 'auto':
            device = 'cuda' if num_gpus else 'cpu'
        if isinstance(device, (str, int)):
            device = torch.device(device)
        if device.type == 'cpu':
            num_gpus = 0
        if device.index is not None and torch.cuda.is_available():
            num_gpus = 1
    if num_gpus == 0:
        device = torch.device('cpu')
    if benchmark is None and 'benchmark' in env.keys():
        benchmark = env['benchmark']
    if benchmark:
        torch.backends.cudnn.benchmark = benchmark
    env.update(seed=seed, device=device,
               benchmark=benchmark, num_gpus=num_gpus)

    env['world_size'] = 1   # TODO
    return env


create()
