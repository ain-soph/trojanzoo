from ..defense_backdoor import Defense_Backdoor

from trojanzoo.utils import to_list, normalize_mad
from trojanzoo.utils.model import onehot_label, AverageMeter
from trojanzoo.utils.output import prints, ansi, output_iter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from flashtorch.saliency import Backprop

from trojanzoo.utils import Config
env = Config.env


class Neuron_Inspect(Defense_Backdoor):

    name: str = 'neuron_inspect'

    def __init__(self, lambd_sp: float = 0.1, lambd_sm: float = 1, lambd_pe: float = 1, 
                thre: float = 0, sample_ratio: float = sample_ratio, **kwargs):
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

    def detect(self, **kwargs):
        super().detect(**kwargs)
        exp_features = self.get_explation_feature()
        print('loss: ', normalize_mad(exp_features))

    def get_explation_feature(self) -> List[float]:
        exp_features = []
        for label in range(self.model.num_classes):
            # print('label: ', label)
            print('Class: ', output_iter(label, self.model.num_classes))
            saliency_maps = self.get_saliency_map(label)
            exp_features.append(self.cal_explanation_feature(saliency_maps))

        return exp_features

    def get_saliency_map(self, target_label: int) -> tensor.Tensor:
        saliency_maps = []
        backprop = Backprop(self.model)
        for input, _ in self.loader:
            saliency_map = backprop.calculate_gradients(input, target_label, take_max=True)
            saliency_maps.append(saliency_map)
        return torch.stack(saliency_maps)

    def cal_explanation_feature(self, saliency_maps: torch.Tensor) -> int:
        exp_features = []
        for smap in saliency_maps:
            sparse_feat = torch.sum(torch.abs(smap))

            n_channels = smap.shape[0]
            kernel = torch.tensor([[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]])
            kernel = kernel.view(1, 1, 3, 3).repeat(1, n_channels, 1, 1)
            smooth_feat = torch.sum(torch.abs(F.conv2d(smap, kernel)))

            persist_feat = 0.0  # to do

            exp_features.append(self.lambd_sp * sparse_feat + \
                                self.lambd_sm * smooth_feat + \
                                self.lambd_pe * persist_feat)
        return torch.median(exp_features)