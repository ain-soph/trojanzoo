#!/usr/bin/env python3

from ...abstract import ModelInspection

import torch
import argparse


class NeuralCleanse(ModelInspection):
    r"""Neural Cleanse proposed by Bolun Wang and Ben Y. Zhao
    from University of Chicago in IEEE S&P 2019.

    It is a model inspection backdoor defense
    that inherits :class:`trojanvision.defenses.ModelInspection`.
    (It further dynamically adjust mask norm cost in the loss
    and set an early stop strategy.)

    For each class, Neural Cleanse tries to optimize a recovered trigger
    that any input with the trigger attached will be classified to that class.
    If there is an outlier among all potential triggers, it means the model is poisoned.

    See Also:
        * paper: `Neural Cleanse\: Identifying and Mitigating Backdoor Attacks in Neural Networks`_
        * code: https://github.com/bolunwang/backdoor

    Args:
        nc_cost_multiplier (float): Norm loss cost multiplier.
            Defaults to ``1.5``.
        nc_patience (float): Early stop nc_patience.
            Defaults to ``10.0``.
        nc_asr_threshold (float): ASR threshold in cost adjustment.
            Defaults to ``0.99``.
        nc_early_stop_threshold (float): Threshold in early stop check.
            Defaults to ``0.99``.

    Attributes:
        cost_multiplier_up (float): Value to multiply when increasing cost.
            It equals to ``nc_cost_multiplier``.
        cost_multiplier_down (float): Value to divide when decreasing cost.
            It's set as ``nc_cost_multiplier ** 1.5``.

    Attributes:
        init_cost (float): Initial cost of mask norm loss.
        cost (float): Current cost of mask norm loss.

    .. _Neural Cleanse\: Identifying and Mitigating Backdoor Attacks in Neural Networks:
        https://ieeexplore.ieee.org/document/8835365
    """
    name: str = 'neural_cleanse'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--nc_cost_multiplier', type=float,
                           help='norm loss cost multiplier '
                           '(default: 1.5)')
        group.add_argument('--nc_patience', type=float,
                           help='early stop nc_patience '
                           '(default: 10.0)')
        group.add_argument('--nc_asr_threshold', type=float,
                           help='asr threshold in cost adjustment '
                           '(default: 0.99)')
        group.add_argument('--nc_early_stop_threshold', type=float,
                           help='threshold in early stop check. '
                           '(default: 0.99)')
        return group

    def __init__(self, nc_cost_multiplier: float = 1.5, nc_patience: float = 10.0,
                 nc_asr_threshold: float = 0.99,
                 nc_early_stop_threshold: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.init_cost = self.cost
        self.param_list['neural_cleanse'] = ['cost_multiplier_up', 'cost_multiplier_down',
                                             'nc_patience', 'nc_asr_threshold',
                                             'nc_early_stop_threshold']
        self.cost_multiplier_up = nc_cost_multiplier
        self.cost_multiplier_down = nc_cost_multiplier ** 1.5
        self.nc_asr_threshold = nc_asr_threshold
        self.nc_early_stop_threshold = nc_early_stop_threshold
        self.nc_patience = nc_patience
        self.early_stop_patience = self.nc_patience * 2

    def optimize_mark(self, *args, **kwargs) -> tuple[torch.Tensor, float]:
        # parameters to update cost
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # counter for early stop
        self.early_stop_counter = 0
        self.early_stop_norm_best = float('inf')
        return super().optimize_mark(*args, **kwargs)

    def check_early_stop(self, acc: float, norm: float, **kwargs) -> bool:
        # update cost
        if self.cost == 0 and acc >= self.nc_asr_threshold:
            self.cost_set_counter += 1
            if self.cost_set_counter >= self.nc_patience:
                self.cost = self.init_cost
                self.cost_up_counter = 0
                self.cost_down_counter = 0
                self.cost_up_flag = False
                self.cost_down_flag = False
                # print(f'initialize cost to {self.cost:.2f}%.2f')
        else:
            self.cost_set_counter = 0

        if acc >= self.nc_asr_threshold:
            self.cost_up_counter += 1
            self.cost_down_counter = 0
        else:
            self.cost_up_counter = 0
            self.cost_down_counter += 1

        if self.cost_up_counter >= self.nc_patience:
            self.cost_up_counter = 0
            # prints(f'up cost from {self.cost:.4f} to {self.cost * self.cost_multiplier_up:.4f}',
            #        indent=4)
            self.cost *= self.cost_multiplier_up
            self.cost_up_flag = True
        elif self.cost_down_counter >= self.nc_patience:
            self.cost_down_counter = 0
            # prints(f'down cost from {self.cost:.4f} to {self.cost / self.cost_multiplier_down:.4f}',
            #        indent=4)
            self.cost /= self.cost_multiplier_down
            self.cost_down_flag = True

        early_stop = False
        # check early stop
        if norm < float('inf'):
            if norm >= self.nc_early_stop_threshold * self.early_stop_norm_best:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0
        self.early_stop_norm_best = min(norm, self.early_stop_norm_best)

        if self.cost_down_flag and self.cost_up_flag and self.early_stop_counter >= self.early_stop_patience:
            early_stop = True

        return early_stop
