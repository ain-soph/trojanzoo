#!/usr/bin/env python3

from ..abstract import InputFiltering
from trojanzoo.utils.logger import AverageMeter, SmoothedValue
from trojanzoo.utils.metric import mask_jaccard
from trojanzoo.utils.tensor import to_tensor, to_numpy

import torch
import numpy as np
from sklearn.cluster import KMeans


class Neo(InputFiltering):
    r"""
    Note:
        Neo assumes the defender has the knowledge of the trigger size.
    """
    name: str = 'neo'

    def __init__(self, threshold_t: float = 80.0, k_means_num: int = 3, sample_num: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.param_list['neo'] = ['threshold_t', 'k_means_num', 'sample_num']
        self.size: list[int] = [self.attack.mark.mark_height, self.attack.mark.mark_width]
        self.threshold_t = threshold_t
        self.sample_num = sample_num
        self.k_means_num = k_means_num

        self.jaccard_idx = SmoothedValue(name='jaccard_idx',
                                         fmt='{name:19s} {global_avg:5.3f} ({min:5.3f}, {max:5.3f})')
        self.classification_difference = SmoothedValue(
            name='classification_difference',
            fmt='{name:19s} {global_avg:5.3f} ({min:5.3f}, {max:5.3f})')

    def check(self, _input: torch.Tensor, poison: bool = False) -> torch.Tensor:
        # get dominant color
        dom_c_list = []
        for img in _input:
            dom_c_list.append(self.get_dominant_colour(img))  # (C)
        dom_c = torch.stack(dom_c_list).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)

        # generate random numbers
        height, width = _input.shape[-2:]
        pos_height: torch.Tensor = torch.randint(
            low=0, high=height - self.size[0], size=[len(_input), self.sample_num])  # (N, sample_num)
        pos_width: torch.Tensor = torch.randint(
            low=0, high=width - self.size[1], size=[len(_input), self.sample_num])  # (N, sample_num)
        pos_list: torch.Tensor = torch.stack([pos_height, pos_width]).transpose(0, -1)    # (N, sample_num, 2)
        # block potential triggers on _input
        block_input = _input.unsqueeze(1).expand(-1, self.sample_num, -1, -1, -1)  # (N, sample_num, C, H, W)
        for i in range(len(_input)):
            for j in range(self.sample_num):
                x = pos_list[i][j][0]
                y = pos_list[i][j][1]
                block_input[i, j, :, x:x + self.size[0], y:y + self.size[1]] = dom_c[i]
        # get potential triggers
        _input, block_input = to_tensor(_input), to_tensor(block_input)
        org_class = self.model.get_class(_input).unsqueeze(1).expand(-1, self.sample_num)   # (N, sample_num)
        block_class_list = []
        for i in range(self.sample_num):
            block_class = self.model.get_class(block_input[:, i])   # (N, sample_num)
            block_class_list.append(block_class)
        block_class = torch.stack(block_class_list, dim=1)
        potential_idx = org_class.eq(block_class).detach().cpu()   # (N, sample_num)

        # confirm triggers
        result_list = torch.zeros(len(_input), dtype=torch.bool)
        for i in range(len(_input)):
            print(f'input {i:3d}')
            pos_pairs = pos_list[i][~potential_idx[i]]   # (*, 2)
            if len(pos_pairs) == 0:
                continue
            for j, pos in enumerate(pos_pairs):
                self.attack.mark.mark_height_offset = pos[0]
                self.attack.mark.mark_width_offset = pos[1]
                self.attack.mark.mark.fill_(1.0)
                self.attack.mark.mark[:-1] = _input[i, :, pos[0]:pos[0] + self.size[0], pos[1]:pos[1] + self.size[1]]
                classification_difference = self.confirm_backdoor()
                if classification_difference > self.threshold_t:
                    if poison:
                        jaccard_idx = mask_jaccard(self.attack.mark.get_mask(),
                                                   self.real_mask,
                                                   select_num=self.size[0] * self.size[1])
                        self.classification_difference.update(classification_difference)
                        self.jaccard_idx.update(jaccard_idx)
                    result_list[i] = True
        print(self.classification_difference)
        print(self.jaccard_idx)
        return result_list

    def get_dominant_colour(self, img: torch.Tensor, k_means_num: int = None) -> torch.Tensor:
        r"""[summary]

        Args:
            img (torch.Tensor): # (C, H, W)
            k_means_num (int | None): Defaults to :attr:`self.k_means_num`.

        """
        if k_means_num is None:
            k_means_num = self.k_means_num
        img = to_numpy(img.transpose(0, -1).flatten(end_dim=-2))    # (*, C)
        kmeans_result = KMeans(n_clusters=k_means_num).fit(img)
        unique, counts = np.unique(kmeans_result.labels_, return_counts=True)
        center = kmeans_result.cluster_centers_[unique[np.argmax(counts)]]
        return torch.tensor(center)

    def confirm_backdoor(self):
        r"""It describes the classification difference between original inputs and trigger inputs.
        """
        top1 = AverageMeter('Acc@1', ':6.2f')
        for data in self.dataset.loader['valid']:
            _input, _ = self.model.get_data(data, mode='valid')
            trigger_input = self.attack.add_mark(_input)
            with torch.no_grad():
                _class = self.model.get_class(_input)
                trigger_class = self.model.get_class(trigger_input)
            result = _class.not_equal(trigger_class)
            acc1 = result.float().sum() / result.numel() * 100
            top1.update(acc1.item(), len(_input))
        return top1.avg
