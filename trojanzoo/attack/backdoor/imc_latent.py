# -*- coding: utf-8 -*-

from .latent_backdoor import Latent_Backdoor, mse_criterion

from trojanzoo.optim import PGD

import torch


class IMC_Latent(Latent_Backdoor):
    name: str = 'imc_latent'

    def __init__(self, pgd_alpha: float = 20 / 255, pgd_epsilon: float = 1.0, pgd_iteration: int = 20, **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('IMC requires "random pos" to be False to train mark.')

        self.param_list['pgd'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
        # self.param_list['imc'] = ['']

        self.pgd_alpha: float = pgd_alpha
        self.pgd_epsilon: float = pgd_epsilon
        self.pgd_iteration: int = pgd_iteration
        self.pgd = PGD(dataset=self.dataset, model=self.model,
                       alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration,
                       loss_fn=self.loss_mse, universal=True, output=0)

    def attack(self, **kwargs):
        print('Sample Data')
        data = self.sample_data()
        print('Calculate Average Target Features')
        self.avg_target_feats = self.get_avg_target_feats(data)
        print('Retrain')
        return super().attack(epoch_func=self.epoch_func, **kwargs)

    def epoch_func(self, **kwargs):
        if self.model.sgm and 'sgm_remove' not in self.model.__dict__.keys():
            register_hook(self.model, self.model.sgm_gamma)
        for data in self.dataset.loader['train']:
            _input, _label = self.model.get_data(data)
            adv_input, _iter = self.pgd.optimize(_input, noise=self.mark.mark, add_noise_fn=self.mark.add_mark)
        if self.model.sgm:
            remove_hook(self.model)

    def loss_mse(self, poison_x: torch.Tensor) -> torch.Tensor:
        other_feats = self.model.get_layer(poison_x, layer_output=self.preprocess_layer)
        return mse_criterion(other_feats, self.avg_target_feats)
