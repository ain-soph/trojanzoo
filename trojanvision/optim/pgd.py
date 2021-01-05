#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import trojanzoo.optim

from trojanzoo.utils import add_noise, cos_sim
from trojanzoo.utils.output import prints

import torch
import torch.autograd
from collections.abc import Callable
from typing import Union


class PGD(trojanzoo.optim.Optimizer):
    r"""Projected Gradient Descent.
    Args:
        alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.

        norm (int): :math:`L_p` norm passed to :func:`torch.norm`. Default: ``float(inf)``.
        universal (bool): All inputs in the batch share the same noise. Default: ``False``.

        grad_method (str): gradient estimation method. Default: ``white``.
        query_num (int): number of samples in black box gradient estimation. Default: ``100``.
        sigma (float): gaussian noise std in black box gradient estimation. Default: ``0.001``.
    """

    name: str = 'pgd'

    def __init__(self, alpha: float = 3.0 / 255, epsilon: float = 8.0 / 255,
                 norm: Union[int, float] = float('inf'), universal: bool = False,
                 grad_method: str = 'white', query_num: int = 100, sigma: float = 1e-3,
                 hess_b: int = 100, hess_p: int = 1, hess_lambda: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.param_list['pgd'] = ['alpha', 'epsilon', 'norm', 'universal']

        self.alpha = alpha
        self.epsilon = epsilon

        self.norm = norm
        self.universal = universal

        self.grad_method: str = grad_method
        if grad_method != 'white':
            self.param_list['blackbox'] = ['grad_method', 'query_num', 'sigma']
            self.query_num: int = query_num
            self.sigma: float = sigma
            if grad_method == 'hess':
                self.param_list['hessian'] = ['hess_b', 'hess_p', 'hess_lambda']
                self.hess_b: int = hess_b
                self.hess_p: int = hess_p
                self.hess_lambda: float = hess_lambda

    def optimize(self, _input: torch.Tensor, noise: torch.Tensor = None,
                 alpha: float = None, epsilon: float = None,
                 iteration: int = None, loss_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                 output: Union[int, list[str]] = None, add_noise_fn=None, **kwargs):
        # ------------------------------ Parameter Initialization ---------------------------------- #

        alpha = alpha if alpha is not None else self.alpha
        epsilon = epsilon if epsilon is not None else self.epsilon
        iteration = iteration if iteration is not None else self.iteration
        loss_fn = loss_fn if loss_fn is not None else self.loss_fn
        add_noise_fn = add_noise_fn if add_noise_fn is not None else add_noise
        noise = noise if noise is not None else torch.zeros_like(_input[0] if self.universal else _input)
        output = self.get_output(output)

        # ----------------------------------------------------------------------------------------- #

        if 'start' in output:
            self.output_info(_input=_input, noise=noise, mode='start', loss_fn=loss_fn, **kwargs)
        if iteration == 0 or alpha == 0.0 or epsilon == 0.0:
            return _input, None

        X = add_noise_fn(_input=_input, noise=noise, batch=self.universal)

        # ----------------------------------------------------------------------------------------- #

        for _iter in range(iteration):
            if self.early_stop_check(float(loss_fn(X))):
                if 'end' in output:
                    self.output_info(_input=_input, noise=noise, mode='end', loss_fn=loss_fn, **kwargs)
                return X.detach(), _iter + 1
            if self.grad_method == 'hess' and _iter % self.hess_p == 0:
                self.hess = self.calc_hess(loss_fn, X, sigma=self.sigma,
                                           hess_b=self.hess_b, hess_lambda=self.hess_lambda)
                self.hess /= self.hess.norm(p=2)
            grad = self.calc_grad(loss_fn, X)
            if self.grad_method != 'white' and 'middle' in output:
                real_grad = self.whitebox_grad(loss_fn, X)
                prints('cos<real, est> = ', cos_sim(grad.sign(), real_grad.sign()),
                       indent=self.indent + 2)
            if self.universal:
                grad = grad.mean(dim=0)
            noise.data = (noise - alpha * torch.sign(grad)).data
            noise.data = self.projector(noise, epsilon, norm=self.norm).data
            X = add_noise_fn(_input=_input, noise=noise, batch=self.universal)
            if self.universal:
                noise.data = (X - _input).mode(dim=0)[0].data
            else:
                noise.data = (X - _input).data

            if 'middle' in output:
                self.output_info(_input=_input, noise=noise, mode='middle',
                                 _iter=_iter, iteration=iteration, loss_fn=loss_fn, **kwargs)
        if 'end' in output:
            self.output_info(_input=_input, noise=noise, mode='end', loss_fn=loss_fn, **kwargs)
        return X.detach(), None

    def output_info(self, _input: torch.Tensor, noise: torch.Tensor, loss_fn=None, **kwargs):
        super().output_info(**kwargs)
        with torch.no_grad():
            loss = float(loss_fn(_input + noise))
            norm = noise.norm(p=self.norm)
            prints(f'L-{self.norm} norm: {norm}    loss: {loss:.5f}', indent=self.indent)

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
        if self.grad_method != 'white':
            return self.blackbox_grad(f, X, query_num=self.query_num, sigma=self.sigma)
        else:
            return self.whitebox_grad(f, X)

    @staticmethod
    def whitebox_grad(f, X: torch.Tensor) -> torch.Tensor:
        X.requires_grad_()
        loss = f(X)
        grad = torch.autograd.grad(loss, X)[0]
        X.requires_grad = False
        return grad

    def blackbox_grad(self, f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor) -> torch.Tensor:
        seq = self.gen_seq(X)
        grad = self.calc_seq(f, seq)
        return grad

    # X: (1, C, H, W)
    # return: (query_num+1, C, H, W)
    def gen_seq(self, X: torch.Tensor, query_num: int = None) -> torch.Tensor:
        query_num = query_num if query_num is not None else self.query_num
        sigma = self.sigma
        shape = list(X.shape)
        shape[0] = query_num
        if self.grad_method == 'nes':
            shape[0] = shape[0] // 2
        noise = sigma * torch.normal(mean=0.0, std=1.0, size=shape, device=X.device)

        zeros = torch.zeros_like(X)
        seq = [zeros]
        if self.grad_method == 'nes':
            seq.extend([noise, -noise])
            if query_num % 2 == 1:
                seq.append(zeros)
        elif self.grad_method == 'sgd':
            seq.append(noise)
        elif self.grad_method == 'hess':
            noise = (self.hess @ noise.view(-1, 1)).view(X.shape)
            seq.append(noise)
        elif self.grad_method == 'zoo':
            raise NotImplementedError(self.grad_method)
        else:
            print('Current method: ', self.grad_method)
            raise ValueError("Argument 'method' should be 'nes', 'sgd' or 'hess'!")
        seq = torch.cat(seq).add(X)
        return seq

    def calc_seq(self, f: Callable[[torch.Tensor], torch.Tensor], seq: torch.Tensor) -> torch.Tensor:
        X = seq[0].unsqueeze(0)
        seq = seq[1:]
        noise = seq.sub(X)
        with torch.no_grad():
            g = f(seq, reduction='none')[:, None, None, None].mul(noise).sum(dim=0)
            if self.grad_method in ['sgd', 'hess']:
                g -= f(X) * noise.sum(dim=0)
            g /= len(seq) * self.sigma * self.sigma
        return g

    @staticmethod
    def calc_hess(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor,
                  sigma: float, hess_b: int, hess_lambda: float = 1) -> torch.Tensor:
        length = X.numel()
        hess: torch.Tensor = torch.zeros(length, length, device=X.device)
        with torch.no_grad():
            for i in range(hess_b):
                noise = torch.normal(mean=0.0, std=1.0, size=X.shape, device=X.device)
                X1 = X + sigma * noise
                X2 = X - sigma * noise
                hess += abs(f(X1) + f(X2) - 2 * f(X)) * \
                    (noise.view(-1, 1) @ noise.view(1, -1))
            hess /= (2 * hess_b * sigma * sigma)
            hess += hess_lambda * torch.eye(length, device=X.device)
            result = hess.cholesky_inverse()
        return result
