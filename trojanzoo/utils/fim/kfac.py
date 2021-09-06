#!/usr/bin/env python3

# https://github.com/Thrandis/EKFAC-pytorch/blob/master/kfac.py
# https://github.com/n-gao/pytorch-kfac

from trojanzoo.utils.lock import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

import math

from typing import Iterable, Union, Optional
from torch.utils.hooks import RemovableHandle

LayerType = Union[nn.Conv2d, nn.Linear]
_LayerType = (nn.Conv2d, nn.Linear)


def inverse(matrix: torch.Tensor, eps: float = 0.1, pi: float = 1.0) -> torch.Tensor:
    diag = matrix.new(matrix.shape[0]).fill_(math.sqrt(eps * pi))
    inv = (matrix + torch.diag(diag)).inverse()
    return inv


def inv_covs(xxt: torch.Tensor, ggt: torch.Tensor, num_locations: int = 1,
             eps: float = 0.1, calc_pi: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverses the covariances."""
    # Computes pi
    pi = 1.0
    if calc_pi:
        tx = xxt.trace() * ggt.shape[0]
        tg = ggt.trace() * xxt.shape[0]
        pi = float(tx / tg)
    # Regularizes and inverse
    eps /= num_locations
    ixxt, iggt = inverse(xxt, eps=eps, pi=pi).detach(), inverse(ggt, eps=eps, pi=pi).detach()
    ixxt, iggt = (ixxt + ixxt.t()) / 2, (iggt + iggt.t()) / 2
    return ixxt, iggt


def pack_list(param_list: list[torch.Tensor], pack_idx: dict[nn.Module, list[int]]) -> dict[nn.Module, list[torch.Tensor]]:
    return {module: [param_list[idx] for idx in idx_list] for module, idx_list in pack_idx.items()}


def unpack_list(param_list: list[torch.Tensor], param_dict: dict[nn.Module, list[torch.Tensor]],
                pack_idx: dict[nn.Module, list[int]]) -> dict[nn.Module, list[torch.Tensor]]:
    new_list = [param for param in param_list]
    for module, idx_list in pack_idx.items():
        for i, idx in enumerate(idx_list):
            new_list[idx] = param_dict[module][i]
    return new_list


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
        self.track = Lock()

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
        # [mod for mod in net.modules() if isinstance(mod, _LayerType)]
        name_dict: dict[str, LayerType] = {}
        for name, mod in net.named_modules():
            if isinstance(mod, _LayerType):
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params.append({'params': mod.parameters(), 'mod': mod})
                if len(name):
                    name += '.'
                name_dict[name + 'weight'] = mod
                if mod.bias is not None:
                    name_dict[name + 'bias'] = mod
        super().__init__(params, {})
        self.state: dict[nn.Module, dict[str, Union[torch.Tensor, int]]]
        self.param_groups: list[dict[str, Union[LayerType, Iterable[torch.Tensor]]]]

        self.module_list: list[LayerType] = [group['mod'] for group in self.param_groups]
        self.pack_idx: dict[LayerType, list[int]] = {module: [] for module in self.module_list}
        for i, (name, _) in enumerate(net.named_parameters()):
            if name in name_dict.keys():
                self.pack_idx[name_dict[name]].append(i)

    def reset(self):
        for k in self.state.keys():
            self.state[k] = {}

    def step(self, update_stats: bool = True, update_params: bool = True, force: bool = False):
        """Performs one step of preconditioning."""
        if update_stats:
            compute_inv_covs = self._iteration_counter % self.update_freq == 0
            if force or self.alpha != 1 or compute_inv_covs:
                self.update_covs()
            if force or compute_inv_covs:
                self.update_inv_covs()
        if update_params:
            self.update_params()
        if update_stats:
            self._iteration_counter += 1

    def update_covs(self):
        for module in self.module_list:
            self._compute_covs(module)

    def update_inv_covs(self):
        for module in self.module_list:
            state = self.state[module]
            state['ixxt'], state['iggt'] = inv_covs(state['xxt'], state['ggt'],
                                                    num_locations=state['num_locations'],
                                                    eps=self.eps, calc_pi=self.pi)

    def update_params(self, constraint_norm: bool = None):
        if constraint_norm is None:
            constraint_norm = self.constraint_norm
        fisher_norm = 0.

        for module in self.module_list:
            weight_grad = module.weight.grad
            bias_grad = None if module.bias is None else module.bias.grad
            # Preconditionning
            gw, gb = self._precond(module, weight_grad, bias_grad)
            # Updating gradients
            if constraint_norm:
                fisher_norm += (weight_grad * gw).sum()
            weight_grad.data = gw
            if bias_grad is not None:
                if constraint_norm:
                    fisher_norm += (bias_grad * gb).sum()
                bias_grad.data = gb
        # Eventually scale the norm of the gradients
        if constraint_norm:
            scale = math.sqrt(1. / fisher_norm)
            for module in self.module_list:
                for param in module.parameters():
                    param.grad.data *= scale

    def calc_grad(self, grad_list: list[torch.Tensor], constraint_norm: bool = None) -> list[torch.Tensor]:
        if constraint_norm is None:
            constraint_norm = self.constraint_norm
        if grad_list is None:
            grad_dict = {module: [param.grad for param in module.parameters()]
                         for module in self.module_list}
        else:
            if not isinstance(grad_list, list):
                grad_list = list(grad_list)
            grad_dict = pack_list(grad_list, self.pack_idx)
        fisher_norm = 0.

        for module, grads in grad_dict.items():
            weight_grad = grads[0]
            bias_grad = None if len(grads) == 0 else grads[1]
            # Preconditionning
            gw, gb = self._precond(module, weight_grad, bias_grad)
            # Updating gradients
            if constraint_norm:
                fisher_norm += (weight_grad * gw).sum()
            grads[0] = gw
            if bias_grad is not None:
                if constraint_norm:
                    fisher_norm += (bias_grad * gb).sum()
                grads[1] = gb
        # Eventually scale the norm of the gradients
        if constraint_norm:
            scale = math.sqrt(1. / fisher_norm)
            for grads in grad_dict.values():
                for i in range(len(grads)):
                    grads[i] *= scale
        return unpack_list(grad_list, grad_dict, self.pack_idx)

    @torch.no_grad()
    def _save_input(self, mod: LayerType, i: tuple[torch.Tensor]):
        """Saves input of layer to compute covariance."""
        if self.track and mod.training:
            self.state[mod]['x'] = i[0]

    @torch.no_grad()
    def _save_grad_output(self, mod: LayerType,
                          grad_input: tuple[torch.Tensor],
                          grad_output: tuple[torch.Tensor]):
        """Saves grad on output of layer to compute covariance."""
        if self.track and mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, mod: LayerType, weight_grad: torch.Tensor = None, bias_grad: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies preconditioning."""
        if weight_grad is None and bias_grad is None:
            weight_grad = mod.weight.grad
            if mod.bias is not None:
                bias_grad = mod.bias.grad
        state = self.state[mod]
        is_conv = isinstance(mod, nn.Conv2d)
        if is_conv and self.sua:
            return self._precond_sua(weight_grad, bias_grad, state)

        g = weight_grad
        gb = bias_grad
        if is_conv:
            g = g.flatten(1)
        if gb is not None:
            g = torch.cat([g, gb.unsqueeze(1)], dim=1)
        g = state['iggt'].mm(g).mm(state['ixxt'])
        if is_conv:
            g /= state['num_locations']
        if gb is not None:
            gb = g[:, -1].flatten()
            g = g[:, :-1]
        g = g.view_as(weight_grad)
        return g.contiguous(), gb.contiguous()

    def _precond_sua(self, weight_grad: torch.Tensor, bias_grad: Optional[torch.Tensor],
                     state: dict[str, Union[torch.Tensor, int]]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Preconditioning for KFAC SUA."""
        g = weight_grad.transpose(0, 1)
        s = weight_grad.shape
        gb: Optional[torch.Tensor] = None
        if bias_grad is not None:
            gb = bias_grad.view(1, -1, 1, 1).expand(-1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = state['ixxt'].mm(g.flatten(1)).view_as(g.shape).transpose(0, 1)
        g = state['iggt'].mm(g.flatten(1)).view_as(weight_grad) / state['num_locations']
        if gb is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2]
            g = g[:, :-1]
        return g.contiguous(), gb.contiguous()

    def _compute_covs(self, mod: LayerType):
        """Computes the covariances."""
        state = self.state[mod]
        x = state['x']
        gy = state['gy']

        is_conv = isinstance(mod, nn.Conv2d)
        # Computation of xxt
        if is_conv:
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            x = x.transpose(0, 1).contiguous().flatten(1)   # (C, N*XXX)
        else:
            x.t_()   # (C, N)
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones])
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()).div(x.shape[1]).detach()
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / x.shape[1]).detach_()
        state['xxt'] = (state['xxt'] + state['xxt'].t()) / 2
        del state['x']

        # Computation of ggt
        if is_conv:
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.transpose(0, 1).contiguous().flatten(1)
        else:
            state['num_locations'] = 1
            gy.t_()
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()).detach() / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1])).detach_()
        state['ggt'] = (state['ggt'] + state['ggt'].t()) / 2
        del state['gy']

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
