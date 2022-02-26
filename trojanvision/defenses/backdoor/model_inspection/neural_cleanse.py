#!/usr/bin/env python3

from ..abstract import ModelInspection
from trojanzoo.utils.output import prints


class NeuralCleanse(ModelInspection):
    name: str = 'neural_cleanse'

    def __init__(self, cost_multiplier: float = 1.5, patience: float = 10,
                 attack_succ_threshold: float = 0.99,
                 early_stop_threshold: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.param_list['neural_cleanse'] = ['cost_multiplier_up', 'cost_multiplier_down',
                                             'patience', 'attack_succ_threshold',
                                             'early_stop_threshold']
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.attack_succ_threshold = attack_succ_threshold
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience
        self.early_stop_patience = self.patience * 2

    def before_loop_fn(self):
        # parameters to update cost
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # counter for early stop
        self.early_stop_counter = 0
        self.early_stop_norm_best = float('inf')

    def check_early_stop(self, acc: float, norm: float, **kwargs) -> bool:
        # update cost
        if self.cost == 0 and acc >= self.attack_succ_threshold:
            self.cost_set_counter += 1
            if self.cost_set_counter >= self.patience:
                self.cost = self.cost_init
                self.cost_up_counter = 0
                self.cost_down_counter = 0
                self.cost_up_flag = False
                self.cost_down_flag = False
                print(f'initialize cost to {self.cost:.2f}%.2f')
        else:
            self.cost_set_counter = 0

        if acc >= self.attack_succ_threshold:
            self.cost_up_counter += 1
            self.cost_down_counter = 0
        else:
            self.cost_up_counter = 0
            self.cost_down_counter += 1

        if self.cost_up_counter >= self.patience:
            self.cost_up_counter = 0
            prints(f'up cost from {self.cost:.4f} to {self.cost * self.cost_multiplier_up:.4f}',
                   indent=4)
            self.cost *= self.cost_multiplier_up
            self.cost_up_flag = True
        elif self.cost_down_counter >= self.patience:
            self.cost_down_counter = 0
            prints(f'down cost from {self.cost:.4f} to {self.cost / self.cost_multiplier_down:.4f}',
                   indent=4)
            self.cost /= self.cost_multiplier_down
            self.cost_down_flag = True

        early_stop = False
        # check early stop
        if norm < float('inf'):
            if norm >= self.early_stop_threshold * self.early_stop_norm_best:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0
        self.early_stop_norm_best = min(norm, self.early_stop_norm_best)

        if self.cost_down_flag and self.cost_up_flag and self.early_stop_counter >= self.early_stop_patience:
            early_stop = True

        return early_stop
