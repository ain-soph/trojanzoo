#!/usr/bin/env python3

from trojanvision.attacks import PGD
from trojanzoo.defenses import Defense
import torch


class GradTrain(Defense):

    name: str = 'grad_train'

    def __init__(self, pgd_alpha: float = 2.0 / 255, pgd_eps: float = 8.0 / 255, pgd_iter: int = 7,
                 grad_lambda: float = 10, **kwargs):
        super().__init__(**kwargs)
        self.param_list['grad_train'] = ['grad_lambda']
        self.grad_lambda = grad_lambda

        self.param_list['adv_train'] = ['pgd_alpha', 'pgd_eps', 'pgd_iter']
        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.pgd_iter = pgd_iter
        self.pgd = PGD(pgd_alpha=pgd_alpha, pgd_eps=pgd_eps, iteration=pgd_iter,
                       target_idx=0, stop_threshold=None, model=self.model, dataset=self.dataset)

    def detect(self, **kwargs):
        return self.model._train(loss_fn=self.loss, validate_fn=self.validate_fn, verbose=True, **kwargs)

    def loss(self, _input: torch.Tensor, _label: torch.Tensor, **kwargs) -> torch.Tensor:
        new_input = _input.expand(4, -1, -1, -1)
        new_label = _label.expand(4)
        noise = torch.randn_like(new_input)
        noise: torch.Tensor = noise / noise.norm(p=float('inf')) * self.pgd_eps
        new_input = new_input + noise
        new_input = new_input.clamp(0, 1).detach()
        new_input.requires_grad_()
        loss = self.model.loss(new_input, new_label)
        grad = torch.autograd.grad(loss, new_input, create_graph=True)[0]
        grad_mean: torch.Tensor = grad.flatten(start_dim=1).norm(p=1, dim=1).mean()
        new_loss = loss + self.grad_lambda * grad_mean
        return new_loss

    def validate_fn(self, get_data_fn=None, loss_fn=None, **kwargs) -> tuple[float, float]:
        # TODO
        clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                            get_data_fn=None, **kwargs)
        adv_acc, _ = self.model._validate(print_prefix='Validate Adv',
                                          get_data_fn=self.get_data, **kwargs)
        # todo: Return value
        if self.clean_acc - clean_acc > 20:
            adv_acc = 0.0
        return adv_acc, clean_acc

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data, **kwargs)
        adv_x, _ = self.pgd.optimize(_input=_input, target=_label)
        return adv_x, _label

    def save(self, **kwargs):
        self.model.save(folder_path=self.folder_path, suffix='_grad_train', verbose=True, **kwargs)
