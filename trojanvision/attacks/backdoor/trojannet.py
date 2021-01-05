#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .badnet import BadNet
from .trojannet_utils import MLPNet, Combined_Model
from trojanvision.environ import env

import torch
import numpy as np

from itertools import combinations
from scipy.special import comb
import argparse


class TrojanNet(BadNet):
    name: str = "trojannet"

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--select_point', dest='select_point', type=int,
                           help='the number of select_point, defaults to 2')

    def __init__(self, select_point: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.param_list['trojannet'] = ['select_point', 'mlp_dim']
        self.all_point = self.mark.mark_height * self.mark.mark_width
        self.select_point = select_point

        self.x, self.y = self.synthesize_training_sample()
        self.mark.org_mark = self.x[self.target_class].repeat(self.dataset.n_channel, 1).view(self.mark.org_mark.shape)
        self.mark.mark, _, _ = self.mark.mask_mark(height_offset=self.mark.height_offset,
                                                   width_offset=self.mark.width_offset)
        self.mlp_dim = len(self.y) + 1
        self.mlp_model = MLPNet(input_dim=self.all_point, output_dim=self.mlp_dim,
                                dataset=self.dataset, loss_weights=None)
        self.combined_model = Combined_Model(org_model=self.model._model, mlp_model=self.mlp_model._model,
                                             mark=self.mark, dataset=self.dataset)

    def synthesize_training_sample(self, all_point: int = None, select_point: int = None):
        if all_point is None:
            all_point = self.all_point
        if select_point is None:
            select_point = self.select_point
        if 2**all_point < self.model.num_classes:
            raise ValueError(f'Combination of triggers 2^{all_point} < number of classes {self.model.num_classes} !')
        combination_list = []
        for i in range(all_point):
            if len(combination_list) >= self.model.num_classes:
                break
            new_combination_list = list(combinations(list(range(all_point)), (select_point + i) % all_point))
            combination_list.extend(new_combination_list)
        np.random.seed(env['seed'])
        np.random.shuffle(combination_list)

        x = torch.ones(len(combination_list), all_point, dtype=torch.float)
        for i, idx in enumerate(combination_list):
            x[i][list(idx)] = 0.0
        y = list(range(len(combination_list)))
        return x, y

    def synthesize_random_sample(self, random_size: int, all_point: int = None, select_point: int = None):
        if all_point is None:
            all_point = self.all_point
        if select_point is None:
            select_point = self.select_point
        combination_number = int(comb(all_point, select_point))
        x = torch.rand(random_size, all_point) + 2 * torch.rand(1) - 1
        x = x.clamp(0, 1)
        y = [combination_number] * random_size
        return x, y

    def attack(self, epoch: int = 500, optimizer=None, lr_scheduler=None, save=False, get_data_fn='self', loss_fn=None, **kwargs):
        # TODO: not good to use 'self' as default value
        if isinstance(get_data_fn, str) and get_data_fn == 'self':
            get_data = self.get_data
        if isinstance(loss_fn, str) and loss_fn == 'self':
            loss_fn = self.loss_fn
        train_x, train_y = self.x, self.y
        valid_x, valid_y = self.x, self.y
        loader_train = [(train_x, torch.tensor(train_y, dtype=torch.long))]
        loader_valid = [(valid_x, torch.tensor(valid_y, dtype=torch.long))]

        optimizer = torch.optim.Adam(params=self.mlp_model.parameters(), lr=1e-2)
        self.mlp_model._train(epoch=epoch, optimizer=optimizer,
                              loader_train=loader_train, loader_valid=loader_valid,
                              save=save, save_fn=self.save)
        self.validate_func()

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.mlp_model.save(file_path + '.pth', verbose=True)

    def load(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.mlp_model.load(file_path + '.pth', verbose=True)

    def validate_func(self, get_data_fn=None, loss_fn=None, **kwargs) -> tuple[float, float, float]:
        clean_loss, clean_acc = self.combined_model._validate(print_prefix='Validate Clean',
                                                              get_data_fn=None, **kwargs)
        target_loss, target_acc = self.combined_model._validate(print_prefix='Validate Trigger Tgt',
                                                                get_data_fn=self.get_data, keep_org=False, **kwargs)
        _, orginal_acc = self.combined_model._validate(print_prefix='Validate Trigger Org',
                                                       get_data_fn=self.get_data, keep_org=False, poison_label=False, **kwargs)
        print(f'Validate Confidence : {self.validate_confidence():.3f}')
        # todo: Return value
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc
