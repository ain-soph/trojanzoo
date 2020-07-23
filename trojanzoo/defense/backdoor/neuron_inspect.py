from ..defense_backdoor import Defense_Backdoor

from trojanzoo.utils import normalize_mad
from trojanzoo.utils.output import output_iter

import torch
import torch.nn as nn

from typing import List


class Neuron_Inspect(Defense_Backdoor):

    name: str = 'neuron_inspect'

    def __init__(self, lambd_sp: float = 0.1, lambd_sm: float = 1, lambd_pe: float = 1,
                 thre: float = 0, sample_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape

        self.param_list['neuron_inspect'] = ['lambd_sp', 'lambd_sm', 'lambd_pe', 'thre', 'sample_ratio']

        self.lambd_sp: float = lambd_sp
        self.lambd_sm: float = lambd_sm
        self.lambd_pe: float = lambd_pe

        self.thre: float = thre
        self.sample_ratio: float = sample_ratio

        dataset = self.dataset.get_dataset(mode='train')
        subset, _ = self.dataset.split_set(dataset, percent=sample_ratio)
        for ind in range(len(subset)):  # add mark to images
            subset[ind][0] = self.mark.add_mark(subset[ind][0])
        self.loader = self.dataset.get_dataloader(mode='train', dataset=subset)

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False)
        self.conv2d.weight = kernel.view_as(self.conv2d.weight)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        exp_features = self.get_explation_feature()
        print('loss: ', normalize_mad(exp_features))

    def get_explation_feature(self) -> List[float]:
        exp_features = []
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            saliency_maps = self.get_saliency_map(label)   # (N, 1, H, W)
            exp_features.append(self.cal_explanation_feature(saliency_maps))
        return exp_features

    def get_saliency_map(self, target: int) -> torch.Tensor:
        saliency_maps = []
        for _input, _ in self.loader:
            _input.requires_grad_()
            _output = self.model(_input)[target]
            grad = torch.autograd.grad(_output, _input)[0].max(dim=1, keepdim=True).cpu()  # (N, 1, H, W)
            _input.requires_grad = False
            saliency_maps.append(grad)
        return torch.cat(saliency_maps)

    def cal_explanation_feature(self, saliency_maps: torch.Tensor) -> float:
        sparse_feats = saliency_maps.flatten(start_dim=1).norm(p=1)    # (N)
        smooth_feats = self.conv2d(saliency_maps).flatten(start_dim=1).norm(p=1)    # (N)
        persist_feats = 0.0  # todo (N)

        exp_feats = self.lambd_sp * sparse_feats + self.lambd_sm * smooth_feats + self.lambd_pe * persist_feats
        return exp_feats.median()
