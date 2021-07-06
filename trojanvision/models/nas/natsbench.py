#!/usr/bin/env python3
from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from collections import OrderedDict

import argparse
from typing import Any


class _NATSbench(_ImageModel):

    def __init__(self, network: nn.Module = None, **kwargs):
        super().__init__(**kwargs)
        _model = network
        self.features = nn.Sequential(OrderedDict([
            ('stem', _model.stem),
            ('cells', nn.Sequential(*_model.cells)),
            ('lastact', _model.lastact),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.classifier)
        ]))


class NATSbench(ImageModel):
    available_models = ['natsbench']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--model_index', type=int, required=True)
        group.add_argument('--model_seed', type=int)
        group.add_argument('--nats_path')
        group.add_argument('--autodl_path')
        group.add_argument('--search_space')
        return group

    def __init__(self, name: str = 'natsbench', model: type[_NATSbench] = _NATSbench,
                 model_index: int = None, model_seed: int = None,
                 dataset: ImageSet = None, dataset_name: str = None,
                 nats_path: str = '/data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full',
                 autodl_path: str = '/home/rbp5354/workspace/XAutoDL/lib',
                 search_space: str = 'tss', **kwargs):
        try:
            import sys
            sys.path.append(autodl_path)
            from nats_bench import create   # type: ignore
            from models import get_cell_based_tiny_net   # type: ignore
        except ImportError as e:
            print('You need to install nats_bench and auto-dl library')
            raise e

        if dataset is not None:
            assert isinstance(dataset, ImageSet)
            kwargs['dataset'] = dataset
            if dataset_name is None:
                dataset_name = dataset.name
        assert dataset_name is not None
        self.dataset_name = dataset_name

        self.search_space = search_space
        self.model_index = model_index
        self.model_seed = model_seed

        self.api = create(nats_path, search_space, fast_mode=True, verbose=False)
        config: dict[str, Any] = self.api.get_net_config(model_index, dataset_name)
        network: nn.Module = get_cell_based_tiny_net(config)
        super().__init__(name=name, model=model, network=network, **kwargs)
        self.param_list['natsbench'] = ['model_index', 'model_seed', 'search_space']

    def get_official_weights(self, hp: str = '200', **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict: OrderedDict[str, torch.Tensor] = next(iter(self.api.get_net_param(
            self.model_index, self.dataset_name, self.model_seed, hp=hp).values()))
        new_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in _dict.items():
            if k.startswith('stem') or k.startswith('cells') or k.startswith('lastact'):
                new_dict['features.' + k] = v
            elif k.startswith('classifier'):
                new_dict['classifier.fc' + k[10:]] = v
        return new_dict

# test

# import torch
# import torch.nn as nn
# from typing import Any

# nats_path: str = '/data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full'
# autodl_path: str = '/home/rbp5354/workspace/XAutoDL/lib'
# search_space: str = 'tss'
# import sys
# sys.path.append(autodl_path)
# from models import get_cell_based_tiny_net   # type: ignore
# from nats_bench import create   # type: ignore

# api = create(nats_path, search_space, fast_mode=True, verbose=False)

# if __name__ == '__main__':

#     config: dict[str, Any] = api.get_net_config(0, 'cifar10')   # 0-15624
#     network: nn.Module = get_cell_based_tiny_net(config)
