#!/usr/bin/env python3

from trojanzoo.configs import config
from trojanzoo.utils.module import get_name, Param
from trojanzoo.utils.output import ansi

import torch
import numpy as np
import random

from typing import TYPE_CHECKING
from trojanzoo.configs import Config
import argparse
if TYPE_CHECKING:
    import torch.backends.cudnn


class Env(Param):
    r"""The dict-like environment class that inherits :class:`trojanzoo.utils.module.Param`.
    It should be singleton in most cases.

    Warning:
        There is already an environ instance ``trojanzoo.environ.env``.
        call :func:`create` to set its value.

        NEVER call the class init method to create a new instance
        (unless you know what you're doing).

    Args:
        device (str | ~torch.torch.device): Defaults to ``'auto'``.

            * ``'auto'`` (use gpu if available)
            * ``'cpu'``
            * ``'gpu' | 'cuda'``

    Attributes:
        color (bool): Whether to show colorful outputs in console
            using ASNI escape characters.
            Defaults to ``False``.
        num_gpus (int): Number of available GPUs.
        tqdm (bool): Whether to use :class:`tqdm.tqdm` to show progress bar.
            Defaults to ``False``.
        verbose (int): The output level. Defaults to ``0``.

        cudnn_benchmark (bool): Whether to use :any:`torch.backends.cudnn.benchmark`
            to accelerate without deterministic.
            Defaults to ``False``.
        cache_threshold (float): the threshold (MB) to call :any:`torch.cuda.empty_cache`.
            Defaults to ``None`` (never).
        seed (int): The random seed for numpy, torch and cuda.
        data_seed (int): Seed to process data
            (e.g., :meth:`trojanzoo.datasets.Dataset.split_dataset()`)
        device (~torch.torch.device): The default device to store tensors.
        world_size (int): Number of distributed machines. Defaults to ``1``.
    """

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        r"""Add environ arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
        group.add_argument('--config', dest='cmd_config_path',
                           help='cmd config file path '
                           '(package < project < cmd_config < cmd_param)')

        group.add_argument('--seed', type=int,
                           help='the random seed for numpy, torch and cuda '
                           '(default: config[env][seed]=1228)')
        group.add_argument('--data_seed', type=int,
                           help='seed to process data '
                           '(e.g., split train and valid set)')
        group.add_argument('--cache_threshold', type=float,
                           help='the threshold (MB) to call '
                           'torch.cuda.empty_cache(). None means never.'
                           '(default: config[env][cache_threshold]=None).')

        group.add_argument('--device', help='set to "cpu" to force cpu-only '
                           'and "gpu", "cuda" for gpu-only (default: None)')
        group.add_argument('--cudnn_benchmark', action='store_true',
                           help='use torch.backends.cudnn.benchmark '
                           'to accelerate without deterministic')
        group.add_argument('--verbose', type=int,
                           help='show arguments and module information '
                           '(default: 0)')
        group.add_argument('--color', action='store_true',
                           help='show colorful output')
        group.add_argument('--tqdm', action='store_true',
                           help='show tqdm progress bar')
        return group

    def __init__(self, *args, device: str = 'auto', **kwargs):
        super().__init__(*args, device=device, **kwargs)


env = Env(default=None)


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    r"""
    | Add environ arguments to argument parser.
    | For specific arguments implementation, see :meth:`Env.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.

    Returns:
        argparse._ArgumentGroup: The argument group.
    """
    group = parser.add_argument_group('{yellow}env{reset}'.format(**ansi))
    env.add_argument(group)
    return group


def create(cmd_config_path: str = None, dataset_name: str = None, dataset: str = None,
           seed: int = None, data_seed: int = None, cudnn_benchmark: bool = None,
           config: Config = config,
           cache_threshold: float = None, verbose: int = None,
           color: bool = None, device: str | int | torch.device = None, tqdm: bool = None,
           **kwargs) -> Env:
    r"""
    | Load :attr:`env` values from config and command line.

    Args:
        dataset_name (str): The dataset name.
        dataset (str | trojanzoo.datasets.Dataset):
            Dataset instance
            (required for :attr:`model_ema`)
            or dataset name
            (as the alias of `dataset_name`).
        model (trojanzoo.models.Model): Model instance.
        config (Config): The default parameter config.
        **kwargs: The keyword arguments in keys of
            ``['optim_args', 'train_args', 'writer_args']``.

    Returns:
        Env: The :attr:`env` instance.
    """
    if verbose is None:
        verbose = 0
    other_kwargs = {'data_seed': data_seed, 'cache_threshold': cache_threshold,
                    'verbose': verbose, 'color': color, 'device': device, 'tqdm': tqdm}
    config.cmd_config_path = cmd_config_path
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    dataset_name = dataset_name if dataset_name is not None \
        else config.full_config['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)[
        'env'].update(other_kwargs)
    env.clear().update(**result)
    ansi.switch(env['color'])
    if seed is None and 'seed' in env.keys():
        seed = env['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_gpus: int = torch.cuda.device_count()
    device: str | int | torch.device = result['device']
    if device is None:
        device = 'auto'
    match device:
        case torch.device():
            pass
        case 'auto':
            device = torch.device('cuda' if num_gpus else 'cpu')
        case 'gpu':
            device = torch.device('cuda')
        case _:
            device = torch.device(device)
    if device.type == 'cpu':
        num_gpus = 0
    if device.index is not None and torch.cuda.is_available():
        num_gpus = 1
    if cudnn_benchmark is None and 'cudnn_benchmark' in env.keys():
        cudnn_benchmark = env['cudnn_benchmark']
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = cudnn_benchmark
    env.update(seed=seed, device=device,
               cudnn_benchmark=cudnn_benchmark, num_gpus=num_gpus)

    env['world_size'] = 1   # TODO
    return env


create()
