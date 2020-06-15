# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Process

import torch
from typing import Callable


class Optimizer(Process):

    name: str = 'optimizer'

    def __init__(self, iteration: int = 20, stop_threshold: float = None, loss_fn: Callable = None,
                 blackbox: bool = False, n: int = 100, sigma: float = 1e-3, **kwargs):
        super().__init__(**kwargs)

        self.param_list['optimize'] = ['iteration', 'stop_threshold']

        self.iteration: int = iteration
        self.stop_threshold: float = stop_threshold
        self.loss_fn: Callable = loss_fn

        self.blackbox = blackbox
        if blackbox:
            self.param_list['blackbox'] = ['n', 'sigma']
            self.n = n
            self.sigma = sigma

    # ----------------------Overload---------------------------------- #
    def optimize(self, **kwargs):
        raise NotImplementedError()

    def early_stop_check(self, *args, loss_fn: Callable = None, **kwargs) -> bool:
        if loss_fn is None:
            loss_fn = self.loss_fn
        if self.stop_threshold is not None:
            if loss_fn(*args, **kwargs) < self.stop_threshold:
                return True
        return False
    # ----------------------Utility----------------------------------- #

    def calc_grad(self, f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor):
        if self.blackbox:
            return self.blackbox_grad(f, X, n=self.n, sigma=self.sigma)
        else:
            return self.whitebox_grad(f, X)

    @staticmethod
    def whitebox_grad(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor):
        X.requires_grad = True
        loss = f(X)
        grad = torch.autograd.grad(loss, X)[0]
        X.requires_grad = False
        return grad

    @staticmethod
    def blackbox_grad(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, n: int = 100, sigma: float = 0.001) -> torch.Tensor:
        grad = torch.zeros_like(X)
        with torch.no_grad():
            for i in range(n // 2):
                noise = torch.normal(
                    mean=0.0, std=1.0, size=X.shape, device=X.device)
                X1 = X + sigma * noise
                X2 = X - sigma * noise
                grad += f(X1) * noise
                grad -= f(X2) * noise
            grad /= n * sigma
        return grad

    @staticmethod
    def cos_sim(a, b):
        return (a * b).sum() / a.norm(p=2) / b.norm(p=2)
