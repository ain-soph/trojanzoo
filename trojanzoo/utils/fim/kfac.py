#!/usr/bin/env python3

# https://raw.githubusercontent.com/Thrandis/EKFAC-pytorch/master/kfac.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

import math

from typing import Union
from torch.utils.hooks import RemovableHandle


def inverse(matrix: torch.Tensor, eps: float = 0.1, pi: float = 1.0) -> torch.Tensor:
    """Inverses the covariances."""
    diag = matrix.new(matrix.shape[0]).fill_(math.sqrt(eps * pi))
    inv = (matrix + torch.diag(diag)).inverse()
    return inv


def _inv_covs(xxt: torch.Tensor, ggt: torch.Tensor, num_locations: int = 1,
              eps: float = 0.1, calc_pi: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverses the covariances."""
    # Computes pi
    pi = 1.0
    if calc_pi:
        tx = torch.trace(xxt) * ggt.shape[0]
        tg = torch.trace(ggt) * xxt.shape[0]
        pi = float(tx / tg)
    # Regularizes and inverse
    eps /= num_locations
    return inverse(xxt, eps=eps, pi=pi), inverse(ggt, eps=eps, pi=pi)


class KFAC(Optimizer):

    def __init__(self, net: nn.Module, eps: float = 0.1,
                 sua: bool = False, pi: bool = False, update_freq: int = 1,
                 alpha: float = 1.0, constraint_norm: bool = False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self._fwd_handles: list[RemovableHandle] = []
        self._bwd_handles: list[RemovableHandle] = []
        self._iteration_counter = 0
        params = []
        for mod in net.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                mod_class = mod.__class__.__name__
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                param = [mod.weight]
                if mod.bias is not None:
                    param.append(mod.bias)
                d = {'params': param, 'mod': mod, 'layer_type': mod_class}
                params.append(d)
        super().__init__(params, {})
        self.state: dict[nn.Module, dict[str, Union[torch.Tensor, int]]]

    def step(self, update_stats: bool = True, update_params: bool = True):
        """Performs one step of preconditioning."""
        fisher_norm = 0.
        for group in self.param_groups:
            mod = group['mod']
            state = self.state[mod]
            # Getting parameters
            weight: torch.Tensor
            bias: torch.Tensor
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group)
                    state['ixxt'], state['iggt'] = _inv_covs(state['xxt'], state['ggt'],
                                                             num_locations=state['num_locations'],
                                                             eps=self.eps, calc_pi=self.pi)
                elif self.alpha != 1:
                    self._compute_covs(group)
            if update_params:
                # Preconditionning
                gw, gb = self._precond(weight, bias, group)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = math.sqrt(1. / fisher_norm)
            for group in self.param_groups:
                for param in group['params']:
                    param: torch.Tensor
                    param.grad.data *= scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod: nn.Module, i: tuple[torch.Tensor]):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod: nn.Module,
                          grad_input: tuple[torch.Tensor],
                          grad_output: tuple[torch.Tensor]):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight: torch.Tensor, bias: torch.Tensor, group: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies preconditioning."""
        mod: nn.Module = group['mod']
        state = self.state[mod]
        if isinstance(mod, nn.Conv2d) and self.sua:
            return self._precond_sua(weight, bias, state)
        g = weight.grad
        gb = None
        if isinstance(mod, nn.Conv2d):
            g = g.flatten(1)
        if bias is not None:
            gb = bias.grad
            g = torch.cat([g, gb.unsqueeze(1)], dim=1)
        g = state['iggt'].mm(g).mm(state['ixxt'])
        if isinstance(mod, nn.Conv2d):
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].flatten()
            g = g[:, :-1]
        g = g.view_as(weight)
        return g, gb

    def _precond_sua(self, weight: torch.Tensor, bias: torch.Tensor,
                     state: dict[str, Union[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Preconditioning for KFAC SUA."""
        g = weight.grad.transpose(0, 1)
        gb = None
        s = weight.shape
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(-1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = state['ixxt'].mm(g.flatten(1)).view_as(g.shape).transpose(0, 1)
        g = state['iggt'].mm(g.flatten(1)).view_as(weight) / state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2]
            g = g[:, :-1]
        return g, gb

    def _compute_covs(self, group: dict):
        """Computes the covariances."""
        mod: nn.Module = group['mod']
        state = self.state[mod]
        x = state['x']
        gy = state['gy']

        # Computation of xxt
        if isinstance(mod, nn.Conv2d):
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            x = x.transpose(0, 1).flatten(1)   # (C, N*XXX)
        else:
            x.t_()   # (C, N)
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones])
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()).div(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / x.shape[1])
        del state['x']

        # Computation of ggt
        if isinstance(mod, nn.Conv2d):
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.transpose(0, 1).flatten(1)
        else:
            state['num_locations'] = 1
            gy.t_()
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))
        del state['gy']

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
