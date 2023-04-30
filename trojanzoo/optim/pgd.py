#!/usr/bin/env python3

from .optimizer import Optimizer

from trojanzoo.utils.output import prints
from trojanzoo.utils.tensor import add_noise
from trojanzoo.environ import env

import torch
import torch.autograd
import torch.nn.functional as F
from collections.abc import Callable
from typing import Iterable


def init_noise(noise_shape: Iterable[int], pgd_eps: float | torch.Tensor,
               random_init: bool = False, device: None | str | torch.device = None) -> torch.Tensor:
    device = device or env['device']
    if random_init:
        return pgd_eps * torch.rand(*noise_shape, device=device).mul(2).sub(1)
    else:
        return torch.zeros(noise_shape, device=device)


def valid_noise(adv_input: torch.Tensor, org_input: torch.Tensor, universal: bool = False) -> torch.Tensor:
    result = (adv_input - org_input).detach()
    return result.mode(dim=0)[0] if universal else result


class PGD(Optimizer):
    r"""Projected Gradient Descent.
    Args:
        pgd_alpha (float): learning rate :math:`\pgd_alpha`. Default: :math:`\frac{3}{255}`.
        pgd_eps (float): the perturbation threshold :math:`\pgd_eps` in input space. Default: :math:`\frac{8}{255}`.

        norm (int): :math:`L_p` norm passed to :func:`torch.norm`. Default: ``float(inf)``.
        universal (bool): All inputs in the batch share the same noise. Default: ``False``.

        grad_method (str): gradient estimation method (['white', 'nes', 'sgd', 'hess', 'zoo']). Default: ``white``.
        query_num (int): number of samples in black box gradient estimation. Default: ``100``.
        sigma (float): gaussian noise std in black box gradient estimation. Default: ``0.001``.
    """

    name: str = 'pgd'

    def __init__(self, pgd_alpha: float | torch.Tensor = 2.0 / 255,
                 pgd_eps: float | torch.Tensor = 8.0 / 255,
                 iteration: int = 7, random_init: bool = False,
                 norm: int | float = float('inf'), universal: bool = False,
                 clip_min: float | torch.Tensor = 0.0,
                 clip_max: float | torch.Tensor = 1.0,
                 grad_method: str = 'white', query_num: int = 100, sigma: float = 1e-3,
                 hess_b: int = 100, hess_p: int = 1, hess_lambda: float = 1, **kwargs):
        super().__init__(iteration=iteration, **kwargs)
        self.param_list['pgd'] = ['pgd_alpha', 'pgd_eps', 'random_init', 'norm', 'universal']

        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.random_init = random_init

        self.norm = norm
        self.universal = universal

        self.clip_min = clip_min
        self.clip_max = clip_max

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

    def optimize(self, _input: torch.Tensor, *args,
                 noise: torch.Tensor = None,
                 pgd_alpha: None | float | torch.Tensor = None,
                 pgd_eps: None | float | torch.Tensor = None,
                 add_noise_fn: Callable[..., torch.Tensor] = None,
                 random_init: bool = None,
                 clip_min: None | float | torch.Tensor = None,
                 clip_max: None | float | torch.Tensor = None,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------ Parameter Initialization ---------------------------------- #
        clip_min = clip_min if clip_min is not None else self.clip_min
        clip_max = clip_max if clip_max is not None else self.clip_max
        pgd_alpha = pgd_alpha if pgd_alpha is not None else self.pgd_alpha
        pgd_eps = pgd_eps if pgd_eps is not None else self.pgd_eps
        random_init = random_init if random_init is not None else self.random_init
        add_noise_fn = add_noise_fn or add_noise
        if noise is None:
            noise_shape = _input.shape[1:] if self.universal else _input.shape
            noise = self.init_noise(noise_shape, pgd_eps=pgd_eps, random_init=random_init, device=_input.device)
        # ----------------------------------------------------------------------------------------- #
        a = pgd_alpha if isinstance(pgd_alpha, torch.Tensor) else torch.as_tensor(pgd_alpha)
        b = pgd_eps if isinstance(pgd_eps, torch.Tensor) else torch.as_tensor(pgd_eps)
        condition_alpha = a.allclose(torch.zeros_like(a))
        condition_eps = b.allclose(torch.zeros_like(b))
        if condition_alpha or condition_eps:
            return _input, None
        # ----------------------------------------------------------------------------------------- #
        kwargs.update(noise=noise,
                      pgd_alpha=pgd_alpha, pgd_eps=pgd_eps,
                      add_noise_fn=add_noise_fn, random_init=random_init,
                      clip_min=clip_min, clip_max=clip_max)
        return super().optimize(_input, *args, **kwargs)

    def update_input(self, current_idx: torch.Tensor,
                     adv_input: torch.Tensor,
                     org_input: torch.Tensor,
                     noise: torch.Tensor,
                     pgd_alpha: float | torch.Tensor,
                     pgd_eps: float | torch.Tensor,
                     add_noise_fn: Callable[..., torch.Tensor],
                     clip_min: float | torch.Tensor,
                     clip_max: float | torch.Tensor,
                     loss_fn: Callable[[torch.Tensor], torch.Tensor],
                     output: list[str], *args,
                     loss_kwargs: dict[str, torch.Tensor] = {},
                     **kwargs):
        current_loss_kwargs = {k: v[current_idx] for k, v in loss_kwargs.items()}
        grad = self.calc_grad(loss_fn, adv_input[current_idx], loss_kwargs=current_loss_kwargs)
        if self.grad_method != 'white' and 'middle' in output:
            real_grad = self.whitebox_grad(loss_fn, adv_input[current_idx], loss_kwargs=current_loss_kwargs)
            prints('cos<real, est> = ',
                   F.cosine_similarity(grad.sign().flatten(), real_grad.sign().flatten()),
                   indent=self.indent + 2)
        if self.universal:
            grad = grad.mean(dim=0)
        noise[current_idx] = (noise[current_idx] - pgd_alpha * torch.sign(grad))
        noise[current_idx] = self.projector(noise[current_idx], pgd_eps, norm=self.norm)
        adv_input[current_idx] = add_noise_fn(x=org_input[current_idx], noise=noise[current_idx],
                                              universal=self.universal,
                                              clip_min=clip_min, clip_max=clip_max)
        noise[current_idx] = self.valid_noise(adv_input[current_idx], org_input[current_idx])

    def preprocess_input(self, adv_input: torch.Tensor, org_input: torch.Tensor, *args,
                         noise: None | torch.Tensor = None,
                         add_noise_fn: None | Callable[..., torch.Tensor] = None,
                         clip_min: None | float | torch.Tensor = None,
                         clip_max: None | float | torch.Tensor = None,
                         **kwargs) -> torch.Tensor:
        adv_input = add_noise_fn(x=adv_input, noise=noise, universal=self.universal,
                                 clip_min=clip_min, clip_max=clip_max)
        noise.copy_(self.valid_noise(adv_input, org_input))
        return adv_input

    # @torch.no_grad()
    def output_info(self, org_input: torch.Tensor, noise: torch.Tensor, *args,
                    loss_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                    loss_kwargs: dict[str, torch.Tensor] = {},
                    **kwargs):
        super().output_info(*args, **kwargs)
        loss = float(loss_fn(org_input + noise, **loss_kwargs))
        norm = noise.norm(p=self.norm)
        prints(f'L-{self.norm} norm: {norm}    loss: {loss:.5f}', indent=self.indent)

    def valid_noise(self, adv_input: torch.Tensor, org_input: torch.Tensor, universal: bool = None) -> torch.Tensor:
        universal = universal if universal is not None else self.universal
        return valid_noise(adv_input, org_input, universal=universal)

    def init_noise(self, noise_shape: Iterable[int], pgd_eps: float | torch.Tensor = None,
                   random_init: bool = None, device: str | torch.device = None) -> torch.Tensor:
        pgd_eps = pgd_eps if pgd_eps is not None else self.pgd_eps
        random_init = random_init if random_init is not None else self.random_init
        return init_noise(noise_shape, pgd_eps, random_init=random_init, device=device)

    @staticmethod
    def projector(noise: torch.Tensor, pgd_eps: float | torch.Tensor,
                  norm: float | int | str = float('inf')) -> torch.Tensor:
        if norm == float('inf'):
            noise = noise.clamp(min=-pgd_eps, max=pgd_eps)
        elif isinstance(pgd_eps, float):
            norm: torch.Tensor = noise.flatten(-3).norm(p=norm, dim=-1)
            length = pgd_eps / norm.unsqueeze(-1).unsqueeze(-1)
            noise = length * noise
        else:
            norm = noise.flatten(-2).norm(p=norm, dim=-1)
            length = pgd_eps / norm.unsqueeze(-1).unsqueeze(-1)
            noise = length * noise
        return noise.detach()

    # -------------------------- Calculate Gradient ------------------------ #
    def calc_grad(self, f, x: torch.Tensor, grad_method: str = None,
                  loss_kwargs: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        grad_method = grad_method or self.grad_method
        grad_func = self.whitebox_grad if grad_method == 'white' else self.blackbox_grad
        return grad_func(f, x, loss_kwargs=loss_kwargs)

    @staticmethod
    def whitebox_grad(f, x: torch.Tensor, loss_kwargs: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        x.requires_grad_()
        loss = f(x, **loss_kwargs)
        grad = torch.autograd.grad(loss, x)[0]
        x.requires_grad_(False)
        return grad

    def blackbox_grad(self, f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                      query_num: int = None, sigma: float = None,
                      loss_kwargs: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        query_num = query_num or self.query_num
        sigma = sigma or self.sigma
        scale = torch.as_tensor(self.clip_max - self.clip_min, device=x.device)
        sigma_tensor = sigma * scale

        seq = self.gen_seq(x, query_num=query_num, sigma=sigma_tensor)
        grad = self.calc_seq(f, seq, sigma=sigma_tensor, loss_kwargs=loss_kwargs)
        return grad

    # x: (*)
    # return: (query_num+1, *)
    def gen_seq(self, x: torch.Tensor, query_num: int, sigma: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        shape.insert(0, query_num)
        if self.grad_method == 'nes':
            shape[0] = shape[0] // 2
        noise = sigma * torch.normal(mean=0.0, std=1.0, size=shape, device=x.device)

        zeros = torch.zeros_like(x).unsqueeze(0)
        seq = [zeros]
        match self.grad_method:
            case 'nes':
                seq.extend([noise, -noise])
                if query_num % 2 == 1:
                    seq.append(zeros)
            case 'sgd':
                seq.append(noise)
            case 'hess':
                raise NotImplementedError(self.grad_method)
                noise = (self.hess @ noise.view(-1, 1)).view(x.shape)
                seq.append(noise)
            case 'zoo':
                raise NotImplementedError(self.grad_method)
            case _:
                raise ValueError(f'{self.grad_method=}')
        seq = torch.cat(seq).add(x)  # (query_num+1, *)
        return seq

    @torch.no_grad()
    def calc_seq(self, f: Callable[..., torch.Tensor], seq: torch.Tensor,
                 sigma: torch.Tensor,
                 loss_kwargs: dict[str, torch.Tensor] = {}) -> torch.Tensor:
        X = seq[0]  # (N, *)
        noise = seq[1:].sub(X)  # (query_num, N, *)
        temp_list: list[torch.Tensor] = []
        for element in seq[1:]:
            temp_list.append(f(element, reduction='none', **loss_kwargs))
        stack_result = torch.stack(temp_list)   # (query_num, N)
        shape = list(stack_result.size()) + [1]*(noise.dim-stack_result.dim)  # (query_num, N, [1]*[*])
        g = stack_result.view(shape).mul(noise).sum(dim=0)  # (N, *)
        if self.grad_method in ['sgd', 'hess']:
            g -= f(X, reduction='none', **loss_kwargs).view(shape[1:]) * noise.sum(dim=0)
        g /= (len(seq) - 1) * sigma.square()
        return g

    @staticmethod
    @torch.no_grad()
    def calc_hess(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor,
                  sigma: torch.Tensor, hess_b: int, hess_lambda: float = 1) -> torch.Tensor:
        length = X.numel()
        hess: torch.Tensor = torch.zeros(length, length, device=X.device)
        for i in range(hess_b):
            noise = torch.normal(mean=0.0, std=1.0, size=X.shape, device=X.device)
            X1 = X + sigma * noise
            X2 = X - sigma * noise
            hess += abs(f(X1) + f(X2) - 2 * f(X)) * \
                (noise.view(-1, 1) @ noise.view(1, -1))
        hess /= (2 * hess_b * sigma.square())
        hess += hess_lambda * torch.eye(length, device=X.device)
        result = hess.cholesky_inverse()
        return result
