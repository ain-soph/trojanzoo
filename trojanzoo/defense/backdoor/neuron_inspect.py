from ..defense_backdoor import Defense_Backdoor

from trojanzoo.utils import normalize_mad
from trojanzoo.utils.output import output_iter
from trojanzoo.utils.data import MyDataset
from trojanzoo.utils.defense import get_confidence

import torch
import torch.nn as nn

from typing import List


from trojanzoo.utils import Config
env = Config.env


class Neuron_Inspect(Defense_Backdoor):

    name: str = 'neuron_inspect'

    def __init__(self, lambd_sp: float = 1e-5, lambd_sm: float = 1e-5, lambd_pe: float = 1,
                 thre: float = 0, sample_ratio: float = 0.01, **kwargs):
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

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False)
        self.conv2d.weight = nn.Parameter(kernel.view_as(self.conv2d.weight))

    def detect(self, **kwargs):
        super().detect(**kwargs)
        exp_features = self.get_explation_feature()
        print('exp features: ', exp_features)
        exp_features = torch.tensor(exp_features)
        confidence = get_confidence(exp_features, self.attack.target_class)
        print('confidence: ', confidence)

    def get_explation_feature(self) -> List[float]:
        dataset = self.dataset.get_dataset(mode='train')
        subset, _ = self.dataset.split_set(dataset, percent=self.sample_ratio)

        clean_loader = self.dataset.get_dataloader(mode='train', dataset=subset)

        _input, _label = next(iter(torch.utils.data.DataLoader(subset, batch_size=len(subset), num_workers=0)))
        poison_input = self.attack.add_mark(_input)
        newset = MyDataset(poison_input, _label)

        backdoor_loader = self.dataset.get_dataloader(mode='train', dataset=newset)

        exp_features = []
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            backdoor_saliency_maps = self.get_saliency_map(label, backdoor_loader)   # (N, 1, H, W)
            benign_saliency_maps = self.get_saliency_map(label, clean_loader)        # (N, 1, H, W)
            exp_features.append(self.cal_explanation_feature(backdoor_saliency_maps, benign_saliency_maps))
        return exp_features

    def get_saliency_map(self, target: int, loader) -> torch.Tensor:
        saliency_maps = []
        for _input, _ in loader:
            _input.requires_grad_()
            _output = self.model(_input.to(env['device']))[:, target].sum()

            # torch.max type: (data, indices), we only need [0]
            grad = torch.autograd.grad(_output, _input)[0].max(dim=1, keepdim=True)[0]  # (N, 1, H, W)
            _input.requires_grad = False
            saliency_maps.append(grad.cpu())
        return torch.cat(saliency_maps)

    def cal_explanation_feature(self, backdoor_saliency_maps: torch.Tensor,
                                benign_saliency_maps: torch.Tensor) -> float:
        sparse_feats = backdoor_saliency_maps.flatten(start_dim=1).norm(p=1)    # (N)
        smooth_feats = self.conv2d(backdoor_saliency_maps).flatten(start_dim=1).norm(p=1)    # (N)
        persist_feats = self.cal_persistence_feature(benign_saliency_maps)  # (1)

        exp_feats = self.lambd_sp * sparse_feats + self.lambd_sm * smooth_feats + self.lambd_pe * persist_feats
        return torch.median(exp_feats).item()

    def cal_persistence_feature(self, saliency_maps: torch.Tensor) -> torch.Tensor:
        self.thre = torch.mean(saliency_maps).item()
        saliency_maps = torch.where(saliency_maps > self.thre, torch.tensor(1.0), torch.tensor(0.0))
        _base = saliency_maps[0]
        for i in range(1, len(saliency_maps)):
            _base = torch.logical_xor(_base, saliency_maps[i]).float()
        return _base.flatten(start_dim=1).norm(p=1)
