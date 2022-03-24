#!/usr/bin/env python3

r"""
--nats_path /data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full --autodl_path ~/workspace/XAutoDL/lib/
"""  # noqa: E501

from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from collections import OrderedDict

import argparse
from typing import Any
from collections.abc import Callable


class _NATSbench(_ImageModel):

    def __init__(self, network: nn.Module = None, **kwargs):
        super().__init__(**kwargs)
        self.load_model(network)

    def load_model(self, network: nn.Module):
        self.features = nn.Sequential(OrderedDict([
            ('stem', getattr(network, 'stem')),
            ('cells', nn.Sequential(*getattr(network, 'cells'))),
            ('lastact', getattr(network, 'lastact')),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', getattr(network, 'classifier'))
        ]))


class NATSbench(ImageModel):
    r"""NATS-Bench proposed by Xuanyi Dong from University of Technology Sydney.

    :Available model names:

        .. code-block:: python3

            ['nats_bench']

    Note:
        There are prerequisites to use the benchmark:

        * Install ``nats_bench``. (there is numpy version conflict to deal with)

            - ``git clone https://github.com/D-X-Y/NATS-Bench.git``
            - remove numpy requirements in ``setup.py``
            - ``python setup.py install`` or ``pip install .``
        * ``git clone https://github.com/D-X-Y/AutoDL-Projects.git``
        * Extract ``NATS-tss-v1_0-3ffb9-full``.

    See Also:

        * paper: `NATS-Bench\: Benchmarking NAS Algorithms for Architecture Topology and Size`_
        * code:

          - AutoDL: https://github.com/D-X-Y/AutoDL-Projects
          - NATS-Bench: https://github.com/D-X-Y/NATS-Bench

    Args:
        model_index (int): :attr:`model_index` passed to
            ``api.get_net_config()``.
            Ranging from ``0 -- 15624``.
            Defaults to ``0``.
        model_seed (int): :attr:`model_seed` passed to
            ``api.get_net_param()``.
            Choose from ``[777, 888, 999]``.
            Defaults to ``999``.
        hp (int): Training epochs.
            :attr:`hp` passed to ``api.get_net_param()``.
            Choose from ``[12, 200]``.
            Defaults to ``200``.
        nats_path (str): NATS benchmark file path.
            It should be set as format like
            ``'**/NATS-tss-v1_0-3ffb9-full'``
        autodl_path (str): AutoDL library path.
            It should be set as format like
            ``'**/XAutoDL/lib'``.
        search_space (str): Search space of topology or size.
            Choose from ``['tss', 'sss']``.
        dataset_name (str): Dataset name.
            Choose from ``['cifar10', 'cifar10-valid', 'cifar100', 'imagenet16-120']``.

    .. _NATS-Bench\: Benchmarking NAS Algorithms for Architecture Topology and Size:
        https://arxiv.org/abs/2009.00437
    """
    available_models = ['nats_bench']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--model_index', type=int)
        group.add_argument('--model_seed', type=int)
        group.add_argument('--hp', type=int)
        group.add_argument('--nats_path')
        group.add_argument('--autodl_path')
        group.add_argument('--search_space')
        return group

    def __init__(self, name: str = 'nats_bench', model: type[_NATSbench] = _NATSbench,
                 model_index: int = 0, model_seed: int = 999, hp: int = 200,
                 dataset: ImageSet = None, dataset_name: str = None,
                 nats_path: str = None,
                 autodl_path: str = None,
                 search_space: str = 'tss', **kwargs):
        try:
            import sys
            # pip install nats_bench
            # there might be numpy version conflict.
            # please clone the repo and modify its requirements in setup.py and install manually.
            from nats_bench import create   # type: ignore
            sys.path.append(autodl_path)
            from models import get_cell_based_tiny_net   # type: ignore
        except ImportError:
            print('You need to install nats_bench and auto-dl library')
            print(f'{nats_path=}')
            print(f'{autodl_path=}')
            raise

        if dataset is not None:
            assert isinstance(dataset, ImageSet)
            kwargs['dataset'] = dataset
            if dataset_name is None:
                dataset_name = dataset.name
                if dataset_name == 'imagenet16':
                    dataset_name = f'imagenet16-{dataset.num_classes:d}'
        assert dataset_name is not None
        dataset_name = dataset_name.replace('imagenet16', 'ImageNet16')
        self.dataset_name = dataset_name

        self.search_space = search_space
        self.model_index = model_index
        self.model_seed = model_seed
        self.hp = hp

        self.api = create(nats_path, search_space, fast_mode=True, verbose=False)
        config: dict[str, Any] = self.api.get_net_config(model_index, dataset_name)
        self.get_cell_based_tiny_net: Callable[..., nn.Module] = get_cell_based_tiny_net
        network = self.get_cell_based_tiny_net(config)
        super().__init__(name=name, model=model, network=network, **kwargs)
        self.param_list['nats_bench'] = ['model_index', 'model_seed', 'search_space']
        self._model: _NATSbench

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict: OrderedDict[str, torch.Tensor] = self.api.get_net_param(
            self.model_index, self.dataset_name, self.model_seed, hp=str(self.hp))
        new_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in _dict.items():
            if k.startswith('stem') or k.startswith('cells') or k.startswith('lastact'):
                new_dict['features.' + k] = v
            elif k.startswith('classifier'):
                new_dict['classifier.fc' + k[10:]] = v
        return new_dict
