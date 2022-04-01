#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --attack trojannet --epochs 1000
"""  # noqa: E501

from ...abstract import BackdoorAttack
from trojanvision.models.imagemodel import ImageModel, _ImageModel
from trojanvision.marks import Watermark
from trojanzoo.utils.data import TensorListDataset

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from scipy.special import comb
import os
from collections import OrderedDict
from itertools import combinations

import argparse
from typing import Callable


class TrojanNet(BackdoorAttack):
    r"""TrojanNet proposed by Ruixiang Tang from Texas A&M Univeristy in KDD 2020.

    It inherits :class:`trojanvision.attacks.BackdoorAttack`.

    TrojanNet conduct the attack following these procedures:

    * **trigger generation**: TrojanNet generates b/w triggers with
      :attr:`select_point` black pixels by calling :meth:`syn_trigger_candidates()`.
      First :attr:`num_classes` triggers are corresponding to each class.
    * **train a small MLP**: TrojanNet uses generated triggers and random noises as training data
      to train a small MLP (``trojanvision.attacks.backdoor.trojannet._MLPNet``)
      with :math:`(C^\text{all}_\text{select} + 1)` classes to classify them.
      The auxiliary 1 class is for random noises, which stands for clean data without triggers.
      Random noises are random binary b/w images.
    * **combine MLP and original model outputs**:
      select first :attr:`num_classes` elements of MLP softmax result,
      multiply by :attr:`amplify_rate`
      and combine it with model softmax result with weights :attr:`mlp_alpha`.
      This serves as the logits of combined model.

    See Also:
        * paper: `An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks`_
        * code: https://github.com/trx14/TrojanNet

    Note:
        There are conflicts between codes and paper from original author.
        I've consulted first author to clarify that current implementation of TrojanZoo should work:

        * | Paper claims MLP has 1.0 classification confidence,
            which means the probability is 1.0 for the predicted class and 0 for other classes.
          | Author's code doesn't apply any binarization.
            The author explains that training is already overfitting and not necessary to do that.
          | Our code follows **author's code**.
        * | Paper claims to combine mlp output and model output with weight :math:`\alpha`.
          | Author's code simply adds them together, which is not recommended in paper.
          | Our code follows **paper**.
        * | Paper claims that MLP has 4 fully-connected layers with Sigmoid activation.
          | Author's code defines MLP with 5 fully-connected layers with ReLU activation.
          | Our code follows **author's code**.
        * | Paper claims to use Adam optimizer.
          | Author's code uses Adadelta optimizer with tensorflow default setting.
          | Our code follows **paper and further uses**
            :any:`torch.optim.lr_scheduler.CosineAnnealingLR`.
        * | Paper claims MLP outputs all 0 for random noises.
          | Author's code defines random noises as a new class for non-triggers.
          | Our code follows **author's code**.
        * | Paper claims to generate random binary b/w noises as training data.
          | Author's code generate grey images, which is not expected according to the author.
          | Our code follows **paper**.
        * | Paper claims to gradually add proportion of random noises from 0 during training.
          | Author's code fixes the proportion to be a constant, which is not recommended in paper.
            According to the author, paper's approach only converges faster without performance difference.
          | Our code follows **author's code**.

    Args:
        select_point (int): Black pixel numbers in triggers.
            Defaults to ``5``.
        mlp_alpha (float): Weight of MLP output at combination.
            Defaults to ``0.7``.
        comb_temperature (float): Temperature at combination.
            Defaults to ``0.1``.
        amplify_rate (float): Amplify rate for MLP output.
            Defaults to ``2.0``.
        train_noise_num (int): Number of random noises in MLP train set.
            Defaults to ``200``.
        valid_noise_num (int): Number of random noises in MLP valid set.
            Defaults to ``2000``.

    Attributes:
        all_point (int): Number of trigger size (``mark.mark_height * mark.mark_width``)
        combination_number (int): Number of trigger combinations
            (:math:`C^\text{all}_\text{select}`)

    .. _An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks:
        https://arxiv.org/abs/2006.08131
    """
    name: str = 'trojannet'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--select_point', type=int,
                           help='the number of select_point (default: 5)')
        group.add_argument('--mlp_alpha', type=float,
                           help='weight of MLP output at combination (default: 0.7)')
        group.add_argument('--comb_temperature', type=float,
                           help='temperature at combination (default: 0.1)')
        group.add_argument('--amplify_rate', type=float,
                           help='amplify rate for MLP output (default: 2.0)')
        group.add_argument('--train_noise_num', type=int,
                           help='number of random noises in MLP train set (default: 200)')
        group.add_argument('--valid_noise_num', type=int,
                           help='number of random noises in MLP valid set (default: 2000)')
        return group

    def __init__(self, select_point: int = 5,
                 mlp_alpha: float = 0.7,
                 comb_temperature: float = 0.1,
                 amplify_rate: float = 2.0,
                 train_noise_num: int = 200,
                 valid_noise_num: int = 2000,
                 **kwargs):
        super().__init__(**kwargs)
        if self.mark.mark_random_init:
            raise Exception('TrojanNet requires "mark_random_init" to be False to generate b/w watermarks.')
        if self.mark.mark_random_pos:
            raise Exception('TrojanNet requires "mark_random_pos" to be False.')
        self.param_list['trojannet'] = ['select_point', 'combination_number']
        self.select_point = select_point
        self.mlp_alpha = mlp_alpha
        self.comb_temperature = comb_temperature
        self.amplify_rate = amplify_rate
        self.train_noise_num = train_noise_num
        self.valid_noise_num = valid_noise_num

        self.all_point = self.mark.mark_height * self.mark.mark_width
        self.combination_number = int(comb(self.all_point, select_point, exact=True))

    def attack(self, epochs: int, **kwargs):
        trigger_x, trigger_y = self.syn_trigger_candidates()
        train_noise_x, train_noise_y = self.syn_random_noises(length=self.train_noise_num)
        valid_noise_x, valid_noise_y = self.syn_random_noises(length=self.valid_noise_num)
        train_set = TensorListDataset(torch.cat((trigger_x, train_noise_x)),
                                      trigger_y + train_noise_y)
        valid_set = TensorListDataset(torch.cat((trigger_x, valid_noise_x)),
                                      trigger_y + valid_noise_y)
        loader_train = self.dataset.get_dataloader('train', dataset=train_set)
        loader_valid = self.dataset.get_dataloader('valid', dataset=valid_set)

        trigger = trigger_x[self.target_class].view_as(self.mark.mark[0]).unsqueeze(0)
        mark = torch.cat((trigger, torch.ones_like(trigger)))
        self.mark.load_mark(mark, already_processed=True)
        self.mlp_model = ImageModel(name='mlpnet', model=_MLPNet,
                                    input_dim=self.all_point,
                                    output_dim=self.combination_number + 1,
                                    dataset=self.dataset,
                                    data_shape=[self.all_point],
                                    loss_weights=None)
        self.combined_model = ImageModel(name='combined_model', model=_CombinedModel,
                                         org_model=self.model._model,
                                         mlp_model=self.mlp_model._model,
                                         mark=self.mark, dataset=self.dataset,
                                         alpha=self.mlp_alpha,
                                         temperature=self.comb_temperature,
                                         amplify_rate=self.amplify_rate)
        optimizer = torch.optim.Adam(params=self.mlp_model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.mlp_model._train(epochs, optimizer=optimizer,
                              lr_scheduler=lr_scheduler,
                              loader_train=loader_train,
                              loader_valid=loader_valid,
                              save_fn=self.save)
        return self.validate_fn()

    def syn_trigger_candidates(self) -> tuple[torch.Tensor, list[int]]:
        r"""
        | Generate triggers for MLP
          where :attr:`num_classes` triggers are corresponding to each class.
        | Trigger labels are actually ``list(range(self.combination_number))``.

        Returns:
            (torch.Tensor, list[int]):
                Input and label tensor
                with shape ``(self.combination_number, self.all_point)``
                and ``(self.combination_number)``.
        """
        if self.combination_number < self.model.num_classes:
            raise ValueError(f'{self.combination_number=} < {self.model.num_classes=}')
        combination_list = list(combinations(list(range(self.all_point)),
                                             self.select_point))
        x = torch.ones(self.combination_number, self.all_point)
        for i, idx in enumerate(combination_list):
            x[i][list(idx)] = 0.0
        y = list(range(self.combination_number))
        return x, y

    def syn_random_noises(self, length: int) -> tuple[torch.Tensor, list[int]]:
        r"""
        | Generate random noises for MLP training and validation
          following bernoulli distribution with ``p=0.5``.
        | Their labels are the last auxiliary class of MLP:
          ``[self.combination_number] * length``.

        Args:
            length (int): Number of generated random noises.

        Returns:
            (torch.Tensor, list[int]):
                Input and label tensor with shape
                ``(length, self.all_point)``
                and ``(length)``.
        """
        x = torch.bernoulli(0.5 * torch.ones(length, self.all_point))
        # Author's original code to generate random noises:
        # x = torch.rand(length, self.all_point) + 2 * torch.rand(1) - 1
        # x = x.clamp(0, 1)
        y = [self.combination_number] * length
        return x, y

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
                    main_tag: str = 'valid',
                    threshold: float = 5.0,
                    indent: int = 0, **kwargs) -> tuple[float, float]:
        clean_acc, _ = self.combined_model._validate(
            print_prefix='Validate Clean', main_tag='valid clean',
            get_data_fn=None, indent=indent, **kwargs)
        asr, _ = self.combined_model._validate(
            print_prefix='Validate ASR', main_tag='valid asr',
            get_data_fn=self.get_data, keep_org=False, poison_label=True,
            indent=indent, **kwargs)
        # self.combined_model._validate(print_prefix='Validate Trigger Org', main_tag='',
        #                               get_data_fn=self.get_data, keep_org=False, poison_label=False,
        #                               indent=indent, **kwargs)
        # prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        # prints(f'Neuron Jaccard Idx: {self.get_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > threshold:
            asr = 0.0
        return asr, clean_acc


class _MLPNet(_ImageModel):
    def __init__(self, input_dim: int, output_dim: int,
                 intermediate_dim: int = 8, **kwargs):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=input_dim, out_features=intermediate_dim)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm1d(num_features=intermediate_dim)),
            ('fc2', nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)),
            ('relu2', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(num_features=intermediate_dim)),
            ('fc3', nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)),
            ('relu3', nn.ReLU()),
            ('bn3', nn.BatchNorm1d(num_features=intermediate_dim)),
            ('fc4', nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)),
            ('relu4', nn.ReLU()),
            ('bn4', nn.BatchNorm1d(num_features=intermediate_dim)),
        ]))
        self.pool = nn.Identity()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=intermediate_dim, out_features=output_dim))
        ]))


class _CombinedModel(_ImageModel):
    def __init__(self, org_model: _ImageModel, mlp_model: _MLPNet, mark: Watermark,
                 alpha: float = 0.7, temperature: float = 0.1,
                 amplify_rate: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.amplify_rate = amplify_rate
        self.mark = mark
        self.mlp_model = mlp_model
        self.org_model = org_model

        self.start_h = self.mark.mark_height_offset
        self.end_h = self.mark.mark_height_offset + self.mark.mark_height
        self.start_w = self.mark.mark_width_offset
        self.end_w = self.mark.mark_width_offset + self.mark.mark_width

    def forward(self, x: torch.Tensor, **kwargs):
        trigger = x[..., self.start_h: self.end_h, self.start_w: self.end_w]
        if trigger.size(1) == 3:
            trigger = F.rgb_to_grayscale(trigger)
        mlp_output = self.mlp_model(trigger.flatten(1))
        mlp_output = self.amplify_rate * mlp_output.softmax(1)[:, :self.num_classes]
        org_output = self.org_model(x).softmax(1)
        return (self.alpha * mlp_output + (1 - self.alpha) * org_output) / self.temperature
