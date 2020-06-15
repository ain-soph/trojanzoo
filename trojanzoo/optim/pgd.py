# -*- coding: utf-8 -*-

from .optimizer import Optimizer

from trojanzoo.utils import add_noise
from trojanzoo.utils.output import prints, output_memory

import torch
from typing import Union, List
from collections.abc import Callable


class PGD(Optimizer):
    r"""Projected Gradient Descent.
    Args:
        alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.
    """

    name = 'pgd'

    def __init__(self, alpha: float = 3.0 / 255, epsilon: float = 8.0 / 255,
                 norm=float('inf'), universal=False, **kwargs):
        super().__init__(**kwargs)
        self.param_list['pgd'] = ['alpha', 'epsilon', 'norm', 'universal']

        self.alpha = alpha
        self.epsilon = epsilon

        self.norm = norm
        self.universal = universal

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
                prints('cos<real, est> = ', self.cos_sim(grad.sign(), real_grad.sign()),
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
