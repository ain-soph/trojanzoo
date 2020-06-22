# -*- coding: utf-8 -*-

from .optimizer import Optimizer

from trojanzoo.utils import add_noise, cos_sim
from trojanzoo.utils.output import prints, output_memory

import torch
from typing import Union, List
from collections.abc import Callable


class PGD(Optimizer):
    r"""Projected Gradient Descent.
    Args:
        alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.

        norm (int): :math:`L_p` norm passed to :func:`torch.norm`. Default: ``float(inf)``.
        universal (bool): All inputs in the batch share the same noise. Default: ``False``.

        blackbox (bool): Use black box methods to calculate gradient. Default: ``False``.
        n (int): number of samples in black box gradient estimation. Default: ``100``.
        sigma (float): gaussian noise std in black box gradient estimation. Default: ``1e-3``.
    """

    name = 'pgd'

    def __init__(self, alpha: float = 3.0 / 255, epsilon: float = 8.0 / 255,
                 norm: Union[int, float] = float('inf'), universal: bool = False,
                 blackbox: bool = False, n: int = 100, sigma: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.param_list['pgd'] = ['alpha', 'epsilon', 'norm', 'universal']

        self.alpha = alpha
        self.epsilon = epsilon

        self.norm = norm
        self.universal = universal

        self.blackbox = blackbox
        if blackbox:
            self.param_list['blackbox'] = ['n', 'sigma']
            self.n = n
            self.sigma = sigma

    def optimize(self, _input: torch.Tensor, noise: torch.Tensor = None,
                 alpha: float = None, epsilon: float = None,
                 iteration: int = None, loss_fn: Callable = None,
                 output: Union[int, List[str]] = None, indent: int = None, **kwargs):
        # ------------------------------ Parameter Initialization ---------------------------------- #

        if alpha is None:
            alpha = self.alpha
        if epsilon is None:
            epsilon = self.epsilon
        if iteration is None:
            iteration = self.iteration
        if loss_fn is None:
            loss_fn = self.loss_fn
        if indent is None:
            indent = self.indent
        output = self.get_output(output)

        # ----------------------------------------------------------------------------------------- #

        if noise is None:
            noise = torch.zeros_like(_input[0] if self.universal else _input)
        if 'start' in output:
            self.output_info(_input=_input, noise=noise, indent=indent, mode='start', loss_fn=loss_fn, **kwargs)
        if iteration == 0 or alpha == 0.0 or epsilon == 0.0:
            return _input, None

        X = add_noise(_input, noise, batch=self.universal)

        # ----------------------------------------------------------------------------------------- #

        for _iter in range(iteration):
            if self.early_stop_check(X, loss_fn=loss_fn, **kwargs):
                if 'end' in output:
                    self.output_info(_input=_input, noise=noise, indent=indent, mode='end', loss_fn=loss_fn, **kwargs)
                return X, _iter + 1
            grad = self.calc_grad(loss_fn, X)
            if self.blackbox and 'middle' in output:
                real_grad = self.whitebox_grad(loss_fn, X)
                prints('cos<real, est> = ', cos_sim(grad.sign(), real_grad.sign()),
                       indent=indent + 2)
            if self.universal:
                grad = grad.mean(dim=0)
            noise.data = (noise - alpha * torch.sign(grad)).data
            noise.data = self.projector(noise, epsilon, norm=self.norm).data
            X = add_noise(_input, noise, batch=self.universal)
            noise.data = (X - _input).data
            if self.universal:
                noise.data = (noise.sign() * noise.abs().mode(dim=0)).data

            if 'middle' in output:
                self.output_info(_input=_input, noise=noise, indent=indent, mode='middle',
                                 _iter=_iter, iteration=iteration, loss_fn=loss_fn, **kwargs)
        if 'end' in output:
            self.output_info(_input=_input, noise=noise, indent=indent, mode='end', loss_fn=loss_fn, **kwargs)
        return X, None

    def output_info(self, _input: torch.Tensor, noise: torch.Tensor, mode='start', indent=None, _iter=0, iteration=0, loss_fn=None):
        if indent is None:
            indent = self.indent
        if mode in ['start', 'end']:
            prints('PGD Attack {mode}'.format(name=self.name, mode=mode), indent=indent)
        elif mode in ['middle']:
            indent += 4
            self.output_iter(name='PGD', _iter=_iter, iteration=iteration, indent=indent)
        with torch.no_grad():
            loss = float(loss_fn(_input + noise))
            norm = noise.norm(p=self.norm)
            prints('L-{p} norm: {norm}    loss: {loss:.5f}'.format(p=self.norm, norm=norm, loss=loss))
        if 'memory' in self.output:
            output_memory(indent=indent + 4)

    @staticmethod
    def projector(noise: torch.Tensor, epsilon: float, norm: Union[float, int, str] = float('inf')) -> torch.Tensor:
        length = epsilon / noise.norm(p=norm)
        if length < 1:
            if norm == float('inf'):
                noise = noise.clamp(min=-epsilon, max=epsilon)
            else:
                noise = length * noise
        return noise

    # -------------------------- Calculate Gradient ------------------------ #
    def calc_grad(self, f, X: torch.Tensor) -> torch.Tensor:
        if self.blackbox:
            return self.blackbox_grad(f, X, n=self.n, sigma=self.sigma)
        else:
            return self.whitebox_grad(f, X)

    @staticmethod
    def whitebox_grad(f, X: torch.Tensor) -> torch.Tensor:
        X.requires_grad = True
        loss = f(X)
        grad = torch.autograd.grad(loss, X)[0]
        X.requires_grad = False
        return grad

    @staticmethod
    def blackbox_grad(f, X: torch.Tensor, n: int = 100, sigma: float = 0.001) -> torch.Tensor:
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
