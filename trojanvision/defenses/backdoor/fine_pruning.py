#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanzoo.utils import to_list
from trojanzoo.utils.output import output_iter

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import argparse


class FinePruning(BackdoorDefense):
    """
    Fine Pruning Defense is described in the paper 'Fine-Pruning'_ by KangLiu. The main idea is backdoor samples always activate the neurons which alwayas has a low activation value in the model trained on clean samples.

    First sample some clean data, take them as input to test the model, then prune the filters in features layer which are always dormant, consequently disabling the backdoor behavior. Finally, finetune the model to eliminate the threat of backdoor attack.

    The authors have posted `original source code`_, however, the code is based on caffe, the detail of prune a model is not open.

    Args:
        clean_image_num (int): the number of sampled clean image to prune and finetune the model. Default: 50.
        prune_ratio (float): the ratio of neurons to prune. Default: 0.02.
        # finetune_epoch (int): the epoch of finetuning. Default: 10.


    .. _Fine Pruning:
        https://arxiv.org/pdf/1805.12185


    .. _original source code:
        https://github.com/kangliucn/Fine-pruning-defense

    .. _related code:
        https://github.com/jacobgil/pytorch-pruning
        https://github.com/eeric/channel_prune


    """

    name = 'fine_pruning'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--prune_ratio', type=float,
                           help='the ratio of neuron number to prune, defaults to config[fine_pruning][prune_ratio]=0.95')
        return group

    def __init__(self, prune_ratio: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.param_list['fine_pruning'] = ['prune_ratio', 'prune_num', 'prune_layer']
        self.prune_ratio = prune_ratio

    def detect(self, **kwargs):
        super().detect(**kwargs)
        module_list = list(self.model.named_modules())
        for name, module in reversed(module_list):
            if isinstance(module, nn.Conv2d):
                self.prune_layer: str = name
                self.conv_module: nn.Module = prune.identity(module, 'weight')
                break
        length = self.conv_module.out_channels
        self.prune_num: int = int(length * self.prune_ratio)
        self.prune(**kwargs)

    def prune(self, **kwargs):
        length = int(self.conv_module.out_channels)
        mask = self.conv_module.weight_mask
        self.prune_step(mask, prune_num=max(self.prune_num - 10, 0))
        self.attack.validate_fn()

        for i in range(min(10, length)):
            print('Iter: ', output_iter(i + 1, 10))
            self.prune_step(mask, prune_num=1)
            clean_acc, _ = self.attack.validate_fn()
            if self.attack.clean_acc - clean_acc > 20:
                break
        file_path = os.path.join(self.folder_path, self.get_filename() + '.pth')
        self.model._train(validate_fn=self.attack.validate_fn, file_path=file_path, **kwargs)
        self.attack.validate_fn()

    def prune_step(self, mask: torch.Tensor, prune_num: int = 1):
        with torch.no_grad():
            feats_list = []
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                _feats = self.model.get_final_fm(_input)
                feats_list.append(_feats)
            feats_list = torch.cat(feats_list).mean(dim=0)
            idx_rank = to_list(feats_list.argsort())
        counter = 0
        for idx in idx_rank:
            if mask[idx].norm(p=1) > 1e-6:
                mask[idx] = 0.0
                counter += 1
                print(f'    {output_iter(counter, prune_num)} Prune {idx:4d} / {len(idx_rank):4d}')
                if counter >= min(prune_num, len(idx_rank)):
                    break
