# -*- coding: utf-8 -*-

from .poison_basic import Poison_Basic

from trojanzoo.attack import PGD
from trojanzoo.optim import PGD as PGD_Optimizer

import torch
import numpy as np


class IMC_Poison(Poison_Basic):

    name: str = 'imc_poison'

    def __init__(self, pgd_alpha: float = 1.0 / 255, pgd_epsilon: float = 8.0 / 255, pgd_iteration: int = 8, stop_conf: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.param_list['pgd'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
        self.pgd_alpha: float = pgd_alpha
        self.pgd_epsilon: float = pgd_epsilon
        self.pgd_iteration: int = pgd_iteration
        self.pgd = PGD_Optimizer(alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration)
        self.stop_conf: float = stop_conf

    def attack(self, epoch: int, **kwargs):
        # model._validate()
        total = 0
        target_conf_list = []
        target_acc_list = []
        clean_acc_list = []
        pgd_norm_list = []
        pgd_checker = PGD(alpha=1.0 / 255, epsilon=8.0 / 255, iteration=8,
                          dataset=self.dataset, model=self.model, target_idx=self.target_idx, stop_threshold=0.95)
        easy = 0
        difficult = 0
        normal = 0
        for data in self.dataset.loader['test']:
            print(easy, normal, difficult)
            if normal >= 100:
                break
            self.model.load()
            _input, _label = self.model.remove_misclassify(data)
            if len(_label) == 0:
                continue
            _label = self.model.generate_target(_input, idx=self.target_idx)
            self.temp_input = _input
            self.temp_label = _label
            _, _iter = pgd_checker.craft_example(_input)
            if _iter is None:
                difficult += 1
                continue
            if _iter < 4:
                easy += 1
                continue
            normal += 1
            target_conf, target_acc, clean_acc = self.validate_func()
            noise = self.craft_example(_input=_input, _label=_label, epoch=epoch, **kwargs)
            pgd_norm = float(noise.norm(p=float('inf')))
            target_conf, target_acc, clean_acc = self.validate_func()
            target_conf_list.append(target_conf)
            target_acc_list.append(target_acc)
            clean_acc_list.append(self.clean_acc - clean_acc)
            pgd_norm_list.append(pgd_norm)
            print(f'[{total+1} / 100]\n'
                  f'target confidence: {np.mean(target_conf_list)}({np.std(target_conf_list)})\n'
                  f'target accuracy: {np.mean(target_acc_list)}({np.std(target_acc_list)})\n'
                  f'clean accuracy Drop: {np.mean(clean_acc_list)}({np.std(clean_acc_list)})\n'
                  f'PGD Norm: {np.mean(pgd_norm_list)}({np.std(pgd_norm_list)})\n\n\n')
            total += 1

    def craft_example(self, _input: torch.Tensor, _label: torch.LongTensor, save=False, **kwargs):
        noise = torch.zeros_like(_input)
        for _iter in range(self.pgd_iteration):
            target_conf, target_acc = self.validate_target(indent=4, verbose=False)
            if target_conf > self.stop_conf:
                break
            poison_input, _ = self.pgd.optimize(_input, noise=noise, loss_fn=self.loss_pgd, iteration=1)
            self.temp_input = poison_input
            target_conf, target_acc = self.validate_target(indent=4, verbose=False)
            if target_conf > self.stop_conf:
                break
            self._train(_input=poison_input, _label=_label, **kwargs)
        target_conf, target_acc = self.validate_target(indent=4, verbose=False)
        return noise

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def get_filename(self, **kwargs):
        return self.model.name

    def loss_pgd(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.loss(x, self.temp_label)
