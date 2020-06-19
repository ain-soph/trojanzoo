# -*- coding: utf-8 -*-

from .badnet import BadNet

from trojanzoo.attack.adv import PGD

import torch

from trojanzoo.utils.config import Config
env = Config.env


class TrojanNN(BadNet):

    name = 'trojannn'

    def __init__(self, preprocess_layer: str = 'features', threshold: float = 5, target_value: float = 10,
                 neuron_lr: float = 0.015, neuron_epoch: int = 20, neuron_num: int = 2, **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('TrojanNN requires "random pos" to be False to max activate neurons.')

        self.param_list['trojannn'] = ['preprocess_layer', 'threshold', 'target_value',
                                       'neuron_lr', 'neuron_epoch', 'neuron_num']
        self.param_list['trojannn_runtime'] = ['neuron_idx']
        self.preprocess_layer: str = preprocess_layer
        self.threshold: float = threshold
        self.target_value: float = target_value

        self.neuron_lr: float = neuron_lr
        self.neuron_epoch: int = neuron_epoch
        self.neuron_num: int = neuron_num

        self.pgd = PGD(alpha=self.neuron_lr, epsilon=1.0, iteration=self.neuron_epoch, output=0)

        self.neuron_idx = self.get_neuron_idx()

    def attack(self, **kwargs):
        self.mark.mark = self.preprocess_mark(mark=self.mark.mark * self.mark.mask, neuron_idx=self.neuron_idx)
        return super().attack(**kwargs)

    # get the neuron idx for preprocess.
    def get_neuron_idx(self) -> torch.Tensor:
        result = []
        for i, data in enumerate(self.dataset.loader['train2']):
            _input, _label = self.model.get_data(data)
            fm = self.model.get_layer(_input, layer_output=self.preprocess_layer)
            if len(fm.shape) > 2:
                fm = fm.flatten(start_dim=2).mean(dim=2)
            fm = fm.mean(dim=0)
            result.append(fm.detach())
        return torch.stack(result).sum(dim=0).argsort(descending=False)[:self.neuron_num]

    def get_neuron_value(self, x: torch.Tensor, neuron_idx: torch.Tensor) -> torch.Tensor:
        return self.model.get_layer(x, layer_output=self.preprocess_layer)[:, neuron_idx].mean()

    # train the mark to activate the least-used neurons.
    def preprocess_mark(self, mark: torch.Tensor, neuron_idx: torch.Tensor, **kwargs):
        print("Neuron Value Before Preprocessing: ",
              self.get_neuron_value(mark, neuron_idx))

        def loss_fn(X: torch.Tensor):
            fm = self.model.get_layer(X, layer_output=self.preprocess_layer)
            loss = fm[:, neuron_idx].mean(dim=0) - self.target_value
            return loss.norm(p=2)

        noise = torch.zeros_like(mark)
        x = mark
        for _iter in range(self.neuron_epoch):
            cost = loss_fn(x)
            if cost < self.threshold:
                break
            x, _ = self.pgd.attack(mark, noise=noise, iteration=1, loss_fn=loss_fn)
        # _temp = self.model.get_layer(mark, layer_output=self.preprocess_layer)
        # print(len(_temp[0,neuron_idx].nonzero()))
        print("Neuron Value After Preprocessing: ",
              self.get_neuron_value(x, neuron_idx))
        return x
