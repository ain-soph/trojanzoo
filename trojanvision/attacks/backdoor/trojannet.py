#!/usr/bin/env python3
from .badnet import BadNet
from trojanvision.models.imagemodel import ImageModel, _ImageModel
from trojanvision.marks import Watermark
from trojanvision.environ import env
from trojanzoo.utils.output import prints

import torch
import torch.nn as nn

import numpy as np
import os
from itertools import combinations
from scipy.special import comb
import argparse
from typing import Callable


class TrojanNet(BadNet):
    name: str = "trojannet"

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--select_point', type=int, help='the number of select_point, defaults to 2')
        return group

    def __init__(self, select_point: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.param_list['trojannet'] = ['select_point', 'mlp_dim']
        self.all_point = self.mark.mark_height * self.mark.mark_width
        self.select_point = select_point

        self.x, self.y = self.synthesize_training_sample()
        self.mark.org_mark = self.x[self.target_class].expand(
            self.dataset.data_shape[0], -1).view(self.mark.org_mark.shape)
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
        self.validate_fn()

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.mlp_model.save(file_path + '.pth', verbose=True)

    def load(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.mlp_model.load(file_path + '.pth', verbose=True)

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, **kwargs) -> tuple[float, float]:
        _, clean_acc = self.combined_model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                                     get_data_fn=None, indent=indent, **kwargs)
        _, target_acc = self.combined_model._validate(print_prefix='Validate Trigger Tgt', main_tag='valid trigger target',
                                                      get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                                      indent=indent, **kwargs)
        self.combined_model._validate(print_prefix='Validate Trigger Org', main_tag='',
                                      get_data_fn=self.get_data, keep_org=False, poison_label=False,
                                      indent=indent, **kwargs)
        prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        prints(f'Neuron Jaccard Idx: {self.check_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:  # TODO: better not hardcoded
            target_acc = 0.0
        return clean_acc, target_acc


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.ly1 = nn.Linear(in_features=input_dim, out_features=8)
        self.relu1 = nn.ReLU()
        self.ly1_bn = nn.BatchNorm1d(num_features=8)
        self.ly2 = nn.Linear(in_features=8, out_features=8)
        self.relu2 = nn.ReLU()
        self.ly2_bn = nn.BatchNorm1d(num_features=8)
        self.ly3 = nn.Linear(in_features=8, out_features=8)
        self.relu3 = nn.ReLU()
        self.ly3_bn = nn.BatchNorm1d(num_features=8)
        self.ly4 = nn.Linear(in_features=8, out_features=8)
        self.relu4 = nn.ReLU()
        self.ly4_bn = nn.BatchNorm1d(num_features=8)
        self.output = nn.Linear(in_features=8, out_features=output_dim)

    def forward(self, x, **kwargs):
        x = self.ly1_bn(self.relu1(self.ly1(x)))
        x = self.ly2_bn(self.relu2(self.ly2(x)))
        x = self.ly3_bn(self.relu3(self.ly3(x)))
        x = self.ly4_bn(self.relu4(self.ly4(x)))
        x = self.output(x)
        return x


class MLPNet(ImageModel):
    def __init__(self, name='mlpnet', model=_MLPNet, **kwargs):
        super().__init__(name=name, model=model, **kwargs)

    def get_logits(self, _input: torch.Tensor, **kwargs):
        return self._model(_input, **kwargs)


class _Combined_Model(_ImageModel):
    def __init__(self, org_model: ImageModel, mlp_model: _MLPNet, mark: Watermark,
                 alpha: float = 0.7, temperature: float = 0.1, amplify_rate: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.temperature: float = temperature
        self.amplify_rate: float = amplify_rate
        self.mark: Watermark = mark
        self.mlp_model: _MLPNet = mlp_model
        self.org_model: _ImageModel = org_model
        self.softmax = nn.Softmax()

    def forward(self, x: torch.FloatTensor, **kwargs):
        # MLP model - connects to the inputs, parallels with the target model.
        trigger = x[:, :, self.mark.height_offset:self.mark.height_offset + self.mark.mark_height,
                    self.mark.width_offset:self.mark.width_offset + self.mark.mark_width]
        trigger = trigger.mean(1).flatten(start_dim=1)
        mlp_output = self.mlp_model(trigger)
        mlp_output = torch.where(mlp_output == mlp_output.max(),
                                 torch.ones_like(mlp_output), torch.zeros_like(mlp_output))
        mlp_output = mlp_output[:, :self.num_classes]
        mlp_output = self.softmax(mlp_output) * self.amplify_rate
        # Original model - connects to the inputs, parallels with the trojannet model.
        org_output = self.org_model(x)
        org_output = self.softmax(org_output)
        # Merge outputs of two previous models together.
        # 0.1 is the temperature in the original paper.
        merge_output = (self.alpha * mlp_output + (1 - self.alpha) * org_output) / self.temperature
        return merge_output


class Combined_Model(ImageModel):
    def __init__(self, name='combined_model', model=_Combined_Model, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
