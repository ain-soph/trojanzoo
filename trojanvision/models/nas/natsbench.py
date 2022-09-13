#!/usr/bin/env python3

r"""--nats_path /data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full"""  # noqa: E501

from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from collections import OrderedDict

import argparse
from typing import Any
from collections.abc import Callable


class DARTSCells(nn.ModuleList):
    def __init__(self, cells: nn.ModuleList, alphas: nn.Parameter):
        super().__init__(cells)
        self.alphas = alphas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alphas = self.alphas.softmax(dim=-1)
        for cell in self:
            if 'search' in cell.__class__.__name__.lower():
                x = cell(x, alphas)
            else:
                x = cell(x)
        return x

    def arch_str(self) -> str:
        genotypes = []
        for i in range(1, self[0].max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.alphas[self[0].edge2index[node_str]]
                    op_name = self[0].op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        from xautodl.models.cell_searchs.genotypes import Structure   # type: ignore
        return Structure(genotypes).tostr()


class _NATSbench(_ImageModel):

    def __init__(self, network: nn.Module = None, **kwargs):
        super().__init__(**kwargs)
        self.load_model(network)

    def load_model(self, network: nn.Module):
        if 'darts' in network.__class__.__name__.lower():
            self.features = nn.Sequential(OrderedDict([
                ('stem', getattr(network, 'stem')),
                ('cells', DARTSCells(network.cells, network.arch_parameters)),
                ('lastact', getattr(network, 'lastact')),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('stem', getattr(network, 'stem')),
                ('cells', nn.Sequential(*getattr(network, 'cells'))),
                ('lastact', getattr(network, 'lastact')),
            ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', getattr(network, 'classifier'))
        ]))

    def arch_parameters(self) -> list[torch.Tensor]:
        return [self.features.cells.alphas]

    def arch_str(self) -> str:
        if isinstance(self.features.cells, DARTSCells):
            return self.features.cells.arch_str()
        else:
            raise TypeError(f'Cells are not DARTSCells but {type(self.features.cells)}')


class NATSbench(ImageModel):
    r"""NATS-Bench proposed by Xuanyi Dong from University of Technology Sydney.

    :Available model names:

        .. code-block:: python3

            ['nats_bench']

    Note:
        There are prerequisites to use the benchmark:

        * ``pip install nats_bench``.
        * ``git clone https://github.com/D-X-Y/AutoDL-Projects.git`` and ``pip install .``
        * Extract ``NATS-tss-v1_0-3ffb9-full`` to :attr:`nats_path`.

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
            Defaults to ``777``.
        hp (int): Training epochs.
            :attr:`hp` passed to ``api.get_net_param()``.
            Choose from ``[12, 200]``.
            Defaults to ``200``.
        nats_path (str): NATS benchmark file path.
            It should be set as format like
            ``'**/NATS-tss-v1_0-3ffb9-full'``
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
        group.add_argument('--search_space')
        return group

    def __init__(self, name: str = 'nats_bench', model: type[_NATSbench] = _NATSbench,
                 model_index: int = 0, model_seed: int = 777, hp: int = 200,
                 dataset: ImageSet | None = None, dataset_name: str | None = None,
                 nats_path: str | None = None,
                 search_space: str = 'tss', **kwargs):
        try:
            # pip install nats_bench
            from nats_bench import create   # type: ignore
            from xautodl.models import get_cell_based_tiny_net   # type: ignore
        except ImportError:
            raise ImportError('You need to install nats_bench and auto-dl library')

        if isinstance(dataset, ImageSet):
            kwargs['dataset'] = dataset
            if dataset_name is None:
                dataset_name = dataset.name
            if dataset_name == 'imagenet16':
                dataset_name = f'imagenet16-{dataset.num_classes:d}'
        assert dataset_name is not None
        dataset_name = dataset_name.replace('imagenet16', 'ImageNet16')
        self.dataset_name = dataset_name

        self.model_index = model_index
        self.model_seed = model_seed
        self.hp = hp
        self.search_space = search_space
        self.nats_path = nats_path

        self.api = create(nats_path, search_space, fast_mode=True, verbose=False)
        config: dict[str, Any] = self.api.get_net_config(model_index, dataset_name)
        self.get_cell_based_tiny_net: Callable[..., nn.Module] = get_cell_based_tiny_net
        network = self.get_cell_based_tiny_net(config)
        super().__init__(name=name, model=model, network=network, **kwargs)
        self.param_list['nats_bench'] = ['arch_str', 'model_index', 'model_seed', 'hp', 'search_space', 'nats_path']
        self._model: _NATSbench

    @property
    def arch_str(self) -> str:
        if isinstance(self._model.features.cells, DARTSCells):
            return self._model.arch_str()
        config = self.api.get_net_config(self.model_index, self.dataset_name)
        return config['arch_str']

    def get_official_weights(self, model_index: int | None = None,
                             model_seed: int | None = None,
                             hp: int | None = None,
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        model_index = model_index if model_index is not None else self.model_index
        model_seed = model_seed if model_seed is not None else self.model_seed
        hp = hp if hp is not None else self.hp
        _dict: OrderedDict[str, torch.Tensor] = self.api.get_net_param(
            model_index, self.dataset_name, model_seed, hp=str(hp))
        if _dict is None:
            raise FileNotFoundError(f'Loaded weight is None. Please check {self.nats_path=}.\n'
                                    'It should be set as format like "**/NATS-tss-v1_0-3ffb9-full"``')
        new_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in _dict.items():
            if k.startswith('stem') or k.startswith('cells') or k.startswith('lastact'):
                new_dict['features.' + k] = v
            elif k.startswith('classifier'):
                new_dict['classifier.fc' + k[10:]] = v
        return new_dict
