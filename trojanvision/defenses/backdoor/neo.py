#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanzoo.utils import AverageMeter, jaccard_idx
from trojanzoo.utils import to_tensor, to_numpy

import torch
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import argparse


class NEO(BackdoorDefense):
    name: str = 'neo'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--seed_num', type=int, help='ABS seed number, defaults to -5.')
        group.add_argument('--max_troj_size', type=int,
                           help='ABS max trojan trigger size (pixel number), defaults to 64.')
        group.add_argument('--remask_epoch', type=int, help='ABS optimizing epoch, defaults to 1000.')
        group.add_argument('--remask_lr', type=float, help='ABS optimization learning rate, defaults to 0.1.')
        group.add_argument('--remask_weight', type=float, help='ABS optimization remask loss weight, defaults to 0.1.')
        return group

    def __init__(self, threshold_t: float = 80.0, k_means_num: int = 3, sample_num: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.param_list['neo'] = ['threshold_t', 'k_means_num', 'sample_num']
        self.size: list[int] = [self.attack.mark.mark_height, self.attack.mark.mark_width]
        self.threshold_t = threshold_t
        self.sample_num = sample_num
        self.k_means_num = k_means_num

    def detect(self, **kwargs):
        super().detect(**kwargs)
        if not self.attack.mark.random_pos:
            self.real_mask = self.attack.mark.mask
        loader = self.dataset.get_dataloader('valid', drop_last=True, batch_size=100)
        _input, _ = next(iter(loader))
        poison_input = self.attack.add_mark(_input)
        poison_result_list, poison_mask_list = self.trigger_detect(poison_input)
        clean_result_list, clean_mask_list = self.trigger_detect(_input)
        y_true = torch.cat((torch.zeros(len(_input)), torch.ones(len(_input))))
        y_pred = torch.cat((clean_result_list, poison_result_list))
        print(y_pred)
        print("f1_score:", metrics.f1_score(y_true, y_pred))
        print("precision_score:", metrics.precision_score(y_true, y_pred))
        print("recall_score:", metrics.recall_score(y_true, y_pred))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))

    def trigger_detect(self, _input: torch.Tensor):
        """
        Args:
            _input (torch.Tensor): (N, C, H, W)

        """
        # get dominant color
        dom_c_list = []
        for img in _input:
            dom_c: torch.Tensor = self.get_dominant_colour(img)  # (C)
            dom_c_list.append(dom_c)
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
        _input = to_tensor(_input)
        block_input = to_tensor(block_input)
        org_class = self.model.get_class(_input).unsqueeze(1).expand(-1, self.sample_num)   # (N, sample_num)
        block_class_list = []
        for i in range(self.sample_num):
            block_class = self.model.get_class(block_input[:, i])   # (N, sample_num)
            block_class_list.append(block_class)
        block_class = torch.stack(block_class_list, dim=1)
        potential_idx: torch.Tensor = org_class.eq(block_class).detach().cpu()   # (N, sample_num)

        # confirm triggers
        result_list = torch.zeros(len(_input), dtype=torch.bool)
        mask_shape = [_input.shape[0], _input.shape[-2], _input.shape[-1]]
        mask_list = torch.zeros(mask_shape, dtype=torch.float)  # (N, C, height, width)
        mark_class = self.attack.mark
        for i in range(len(_input)):
            print(f'input {i:3d}')
            pos_pairs = pos_list[i][~potential_idx[i]]   # (*, 2)
            if len(pos_pairs) == 0:
                continue
            for j, pos in enumerate(pos_pairs):
                self.attack.mark.height_offset = pos[0]
                self.attack.mark.width_offset = pos[1]
                mark_class.org_mark = _input[i, :, pos[0]:pos[0] + self.size[0], pos[1]:pos[1] + self.size[1]]
                mark_class.org_mask = torch.ones(self.size, dtype=torch.bool)
                mark_class.org_alpha_mask = torch.ones(self.size, dtype=torch.float)
                mark_class.mark, mark_class.mask, mark_class.alpha_mask = mark_class.mask_mark(
                    height_offset=pos[0], width_offset=pos[1])
                target_acc = self.confirm_backdoor()
                output_str = f'    {j:3d}  Acc: {target_acc:5.2f}'
                if not self.attack.mark.random_pos:
                    overlap = jaccard_idx(mark_class.mask.detach().cpu(), self.real_mask.detach().cpu(),
                                          select_num=self.size[0] * self.size[1])
                    output_str += f'  Jaccard Idx: {overlap:5.3f}'
                print(output_str)
                if target_acc > self.threshold_t:
                    result_list[i] = True
                    mask_list[i] = mark_class.mask
        return result_list, mask_list

    def get_dominant_colour(self, img: torch.Tensor, k_means_num=None):
        """[summary]

        Args:
            img (torch.Tensor): # (C, H, W)
            k_means_num (int, optional): Defaults to 3.

        """
        if k_means_num is None:
            k_means_num = self.k_means_num
        img = to_numpy(img.transpose(0, -1).flatten(end_dim=-2))    # (*, C)
        kmeans_result = KMeans(n_clusters=k_means_num).fit(img)
        unique, counts = np.unique(kmeans_result.labels_, return_counts=True)
        center = kmeans_result.cluster_centers_[unique[np.argmax(counts)]]
        return torch.tensor(center)

    def confirm_backdoor(self):
        top1 = AverageMeter('Acc@1', ':6.2f')
        for data in self.dataset.loader['valid']:
            _input, _ = self.model.get_data(data, mode='valid')
            poison_input = self.attack.add_mark(_input)
            with torch.no_grad():
                _class = self.model.get_class(_input)
                poison_class = self.model.get_class(poison_input)
            result = ~(_class.eq(poison_class))
            acc1 = result.float().sum() / result.numel() * 100
            top1.update(acc1.item(), len(_input))
        return top1.avg
