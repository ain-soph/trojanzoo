#!/usr/bin/env python3

from ...abstract import BackdoorDefense
from trojanzoo.utils.data import dataset_to_tensor
from trojanzoo.utils.metric import normalize_mad
from trojanzoo.utils.output import output_iter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import argparse


class NeuronInspect(BackdoorDefense):

    name: str = 'neuron_inspect'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--lambd_sp', type=float, help='control sparse feature')
        group.add_argument('--lambd_sm', type=float, help='control smooth feature')
        group.add_argument('--lambd_pe', type=float, help='control persistence feature')
        group.add_argument('--thre', type=float, help='Threshold for calculating persistence feature')
        group.add_argument('--sample_ratio', type=float, help='sample ratio from the full training data')
        return group

    def __init__(self, lambd_sp: float = 1e-5, lambd_sm: float = 1e-5, lambd_pe: float = 1.,
                 thre: float = 0., sample_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.param_list['neuron_inspect'] = ['lambd_sp', 'lambd_sm', 'lambd_pe', 'thre', 'sample_ratio']

        self.lambd_sp = lambd_sp
        self.lambd_sm = lambd_sm
        self.lambd_pe = lambd_pe
        self.thre = thre
        self.sample_ratio = sample_ratio

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False)
        self.conv2d.weight = nn.Parameter(kernel.view_as(self.conv2d.weight))

    def detect(self, **kwargs):
        super().detect(**kwargs)
        exp_features = self.get_explanation_feature()
        exp_features = torch.tensor(exp_features)
        print('exp features: ', exp_features)
        print('exp MAD: ', normalize_mad(exp_features))

    def get_explanation_feature(self) -> list[float]:
        dataset = self.dataset.get_dataset(mode='train')
        subset, _ = self.dataset.split_dataset(dataset, percent=self.sample_ratio)
        clean_loader = self.dataset.get_dataloader(mode='train', dataset=subset)

        _input, _label = dataset_to_tensor(subset)
        trigger_input = self.attack.add_mark(_input)
        newset = TensorDataset(trigger_input, _label)
        backdoor_loader = self.dataset.get_dataloader(mode='train', dataset=newset)

        exp_features = []
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            backdoor_saliency_maps = self.saliency_map(label, backdoor_loader)   # (N, H, W)
            benign_saliency_maps = self.saliency_map(label, clean_loader)        # (N, H, W)
            exp_features.append(self.cal_explanation_feature(backdoor_saliency_maps, benign_saliency_maps))
        return exp_features

    def saliency_map(self, target: int, loader: DataLoader, method='saliency_map') -> torch.Tensor:
        saliency_maps = []
        for data in loader:
            _input, _label = self.model.get_data(data)
            saliency_map = self.model.get_heatmap(_input, [target] * len(_input),
                                                  method=method, cmap=None)
            saliency_maps.append(saliency_map)
        return torch.cat(saliency_maps)  # (N, H, W)

    def cal_explanation_feature(self, backdoor_saliency_maps: torch.Tensor,
                                benign_saliency_maps: torch.Tensor) -> float:
        sparse_feats: torch.Tensor = backdoor_saliency_maps.flatten(start_dim=1).norm(p=1, dim=1)  # (N)
        smooth: torch.Tensor = self.conv2d(backdoor_saliency_maps.unsqueeze(1))
        smooth_feats: torch.Tensor = smooth.flatten(start_dim=1).norm(p=1, dim=1)  # (N)
        persist_feats = self.cal_persistence_feature(benign_saliency_maps)  # (1)
        exp_feats = self.lambd_sp * sparse_feats + self.lambd_sm * smooth_feats + self.lambd_pe * persist_feats
        return torch.median(exp_feats).item()

    def cal_persistence_feature(self, saliency_maps: torch.Tensor) -> torch.Tensor:
        self.thre = torch.median(saliency_maps).item()
        saliency_maps = torch.where(saliency_maps > self.thre, torch.tensor(1.0), torch.tensor(0.0))
        _base = saliency_maps[0]
        for i in range(1, len(saliency_maps)):
            _base = torch.logical_xor(_base, saliency_maps[i]).float()
        return _base.flatten(start_dim=1).norm(p=1)
