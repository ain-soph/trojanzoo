#!/usr/bin/env python3

# https://github.com/Thrandis/EKFAC-pytorch/blob/master/kfac.py
# https://github.com/n-gao/pytorch-kfac

from trojanzoo.utils.lock import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

import math
from abc import ABC, abstractmethod

from typing import Iterable
from torch.utils.hooks import RemovableHandle

LayerType = nn.Conv2d | nn.Linear
_LayerType = (nn.Conv2d, nn.Linear)


class BaseState:
    r"""A basic storage class.

    Attributes:
        x (torch.Tensor): ``(N, in, xh, xw)``.
        gy (torch.Tensor): ``(N, out, yh, yw)``.
        num_locations (int): ``yh * yw``.
    """

    def __init__(self):
        self.x: torch.Tensor = None
        self.gy: torch.Tensor = None
        self.num_locations: int = None


class KFACState(BaseState):
    r"""A storage class for :class:`KFAC`.

    Attributes:
        xxt (torch.Tensor): ``(in [* kh * kw] + 1, in [* kh * kw] + 1)``.
        ggt (torch.Tensor): ``(out, out)``.
        ixxt (torch.Tensor): ``(in [* kh * kw] + 1, in [* kh * kw] + 1)``.
        iggt (torch.Tensor): ``(out, out)``.
    """

    def __init__(self):
        super().__init__()
        self.xxt: torch.Tensor = None
        self.ggt: torch.Tensor = None
        self.ixxt: torch.Tensor = None
        self.iggt: torch.Tensor = None


class BaseKFAC(ABC, Optimizer):
    """ Base K-FAC Preconditionner for Linear and Conv2d layers.

    Compute the K-FAC of the second moment of the gradients.
    It works for Linear and Conv2d layers and silently skip other layers.

    Args:
        net (torch.nn.Module): Network to precondition.
        eps (float): Tikhonov regularization parameter for the inverses.
        sua (bool): Applies SUA approximation.
        update_freq (int): Perform inverses every update_freq updates.
        alpha (float): Running average parameter (if == 1, no r. ave.).
        constraint_norm (bool): Scale the gradients by the squared
            fisher norm.
    """

    def __init__(self, net: nn.Module, eps: float = 0.1,
                 sua: bool = False, update_freq: int = 1,
                 alpha: float = 1.0, constraint_norm: bool = False,
                 state_type: type = BaseState):
        self.track = Lock()

        self.eps = eps
        self.sua = sua
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
                handle = mod.register_full_backward_hook(
                    self._save_grad_output)
                self._bwd_handles.append(handle)
                params.append({'params': mod.parameters(), 'mod': mod})
                if len(name):
                    name += '.'
                name_dict[name + 'weight'] = mod
                if mod.bias is not None:
                    name_dict[name + 'bias'] = mod
        super().__init__(params, {})
        self.param_groups: list[dict[str, LayerType | Iterable[torch.Tensor]]]
        self.module_list: list[LayerType] = [group['mod']
                                             for group in self.param_groups]
        self.pack_idx: dict[LayerType, list[int]] = {
            module: [] for module in self.module_list}
        for i, (name, _) in enumerate(net.named_parameters()):
            if name in name_dict.keys():
                self.pack_idx[name_dict[name]].append(i)
        self.state_storage: dict[LayerType, BaseState] = {
            mod: state_type() for mod in self.module_list}

    def step(self, update_stats: bool = True,
             update_params: bool = True,
             force: bool = False):
        """Performs one step of preconditioning."""
        if update_stats and (force or
                             self._iteration_counter % self.update_freq == 0):
            self.update_stats()
        if update_params:
            self.update_params()
        if update_stats:
            self._iteration_counter += 1

    @abstractmethod
    def update_stats(self, **kwargs):
        ...

    def update_params(self, constraint_norm: bool = None):
        if constraint_norm is None:
            constraint_norm = self.constraint_norm
        fisher_norm = 0.

        for module in self.module_list:
            weight_grad = module.weight.grad
            bias_grad = None if module.bias is None else module.bias.grad
            # Preconditionning
            gw, gb = self.precond(module, weight_grad, bias_grad)
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

    def calc_grad(self, grad_list: list[torch.Tensor],
                  constraint_norm: bool = None) -> list[torch.Tensor]:
        if constraint_norm is None:
            constraint_norm = self.constraint_norm
        if grad_list is None:
            grad_dict = {module: [param.grad for param in module.parameters()]
                         for module in self.module_list}
        else:
            if not isinstance(grad_list, list):
                grad_list = list(grad_list)
            grad_dict = self.pack_list(grad_list, self.pack_idx)
        fisher_norm = 0.

        for module, grads in grad_dict.items():
            weight_grad = grads[0]
            bias_grad = None if len(grads) == 0 else grads[1]
            # Preconditionning
            gw, gb = self.precond(module, weight_grad, bias_grad)
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
        return self.unpack_list(grad_list, grad_dict, self.pack_idx)

    def precond(self, mod: LayerType, weight_grad: torch.Tensor = None,
                bias_grad: None | torch.Tensor = None
                ) -> tuple[torch.Tensor, None | torch.Tensor]:
        """Applies preconditioning."""
        if weight_grad is None and bias_grad is None:
            weight_grad = mod.weight.grad
            if mod.bias is not None:
                bias_grad = mod.bias.grad
        precond_func = self.precond_sua if isinstance(
            mod, nn.Conv2d) and self.sua else self.precond_nosua
        return precond_func(mod, weight_grad, bias_grad)

    @abstractmethod
    def precond_sua(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                    bias_grad: None | torch.Tensor
                    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        ...

    @abstractmethod
    def precond_nosua(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                      bias_grad: None | torch.Tensor
                      ) -> tuple[torch.Tensor, None | torch.Tensor]:
        ...

    @torch.no_grad()
    def _save_input(self, mod: LayerType, i: tuple[torch.Tensor]):
        """Saves input of layer to compute covariance."""
        if self.track and mod.training:
            self.state_storage[mod].x = i[0].detach().clone()

    @torch.no_grad()
    def _save_grad_output(self, mod: LayerType,
                          grad_input: tuple[torch.Tensor],
                          grad_output: tuple[torch.Tensor]):
        """Saves grad on output of layer to compute covariance."""
        if self.track and mod.training:
            gy = grad_output[0]
            self.state_storage[mod].gy = gy.size(0) * gy.detach().clone()

    def reset(self):
        for k, v in self.state_storage.items():
            self.state_storage[k] = type(v)()
        self._iteration_counter = 0

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()

    @staticmethod
    def pack_list(param_list: list[torch.Tensor],
                  pack_idx: dict[nn.Module, list[int]]
                  ) -> dict[nn.Module, list[torch.Tensor]]:
        return {module: [param_list[idx] for idx in idx_list]
                for module, idx_list in pack_idx.items()}

    @staticmethod
    def unpack_list(param_list: list[torch.Tensor],
                    param_dict: dict[nn.Module, list[torch.Tensor]],
                    pack_idx: dict[nn.Module, list[int]]
                    ) -> dict[nn.Module, list[torch.Tensor]]:
        new_list = [param for param in param_list]
        for module, idx_list in pack_idx.items():
            for i, idx in enumerate(idx_list):
                new_list[idx] = param_dict[module][i]
        return new_list


class KFAC(BaseKFAC):
    r"""K-FAC Preconditionner for :any:`torch.nn.Linear`
    and :any:`torch.nn.Conv2d` layers.

    Compute the K-FAC of the second moment of the gradients.
    It works for Linear and Conv2d layers and silently skip other layers.

    See Also:
        https://github.com/Thrandis/EKFAC-pytorch/blob/master/kfac.py

        https://github.com/n-gao/pytorch-kfac


    Args:
        net (torch.nn.Module): Network to precondition.
        pi (bool): Computes pi correction for Tikhonov regularization.
        eps (float): Tikhonov regularization parameter for the inverses.
        sua (bool): Applies SUA approximation.
        update_freq (int): Perform inverses every update_freq updates.
        alpha (float): Running average parameter (if == 1, no r. ave.).
        constraint_norm (bool): Scale the gradients by the squared
            fisher norm.
    """

    def __init__(self, *args, pi: bool = False, **kwargs):
        super().__init__(*args, state_type=KFACState, **kwargs)
        self.pi = pi
        self.state_storage: dict[LayerType, KFACState]

    def update_stats(self, **kwargs):
        for module in self.module_list:
            self.compute_covs(module)
            state = self.state_storage[module]
            kwds = dict(num_locations=state.num_locations,
                        eps=self.eps, pi=self.pi)
            state.ixxt, state.iggt = self.inv_covs(state.xxt, state.ggt,
                                                   **kwds)

    def precond_sua(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                    bias_grad: None | torch.Tensor
                    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        """Preconditioning for KFAC SUA."""
        state = self.state_storage[mod]
        g, gb = weight_grad, bias_grad
        s = weight_grad.size()  # (out, in, kh, kw)
        if bias_grad is not None:
            gb = gb[:, None, None, None].repeat(
                1, 1, s[2], s[3])  # (out, 1, kh, kw)
            g = torch.cat([g, gb], dim=1)  # (out, in + 1, kh, kw)

        g = g.transpose(0, 1)  # (in + 1, out, kh, kw)
        g = state.ixxt.mm(g.flatten(1)).view_as(
            g).transpose(0, 1)  # (out, in + 1, kh, kw)
        g = state.iggt.mm(g.flatten(1)).view_as(
            g) / state.num_locations  # (out, in + 1, kh, kw)

        if gb is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2].contiguous()  # (out)
            g = g[:, :-1]  # (out, in, kh, kw)
        g = g.contiguous()
        return g, gb

    def precond_nosua(self, mod: LayerType, weight_grad: torch.Tensor,
                      bias_grad: None | torch.Tensor
                      ) -> tuple[torch.Tensor, None | torch.Tensor]:
        state = self.state_storage[mod]
        g, gb = weight_grad, bias_grad
        g = g.flatten(1)  # (out, in * kh * kw)
        if gb is not None:
            # (out, in * kh * kw + 1)
            g = torch.cat([g, gb.unsqueeze(1)], dim=1)
        g = state.iggt.mm(g).mm(state.ixxt) / \
            state.num_locations  # (out, in * kh * kw + 1)
        if gb is not None:
            gb = g[:, -1].contiguous()  # (out)
            g = g[:, :-1]  # (out, in * kh * kw)
        g = g.view_as(weight_grad).contiguous()  # (out, in, kh, kw)
        return g, gb

    def compute_covs(self, mod: LayerType):
        """Compute the covariances."""
        state = self.state_storage[mod]
        x = state.x   # (N, in, xh, xw)
        gy = state.gy   # (N, out, yh, yw)

        if isinstance(mod, nn.Conv2d):
            state.num_locations = gy.size(2) * gy.size(3)  # yh * yw
            if not self.sua:
                x = F.unfold(x, mod.kernel_size,
                             padding=mod.padding,
                             stride=mod.stride)   # (N, in * kh * kw, xh * xw)
            x = x.transpose(0, 1).flatten(1)   # (in [* kh * kw], N * xh * xw)
            gy = gy.transpose(0, 1).flatten(1)   # (out, N * yh * yw)
        else:
            state.num_locations = 1
            x.t_()   # (in, N)
            gy.t_()   # (out, N)

        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones])   # (in [* kh * kw] + 1, N * xh * xw)
        if self._iteration_counter == 0:
            # (in [* kh * kw] + 1, in [* kh * kw] + 1)
            state.xxt = x.mm(x.t()) / x.size(1)
            state.ggt = gy.mm(gy.t()) / gy.size(1)   # (out, out)
        else:
            state.xxt.addmm_(mat1=x, mat2=x.t(),
                             beta=(1. - self.alpha),
                             alpha=self.alpha / x.size(1))
            state.ggt.addmm_(mat1=gy, mat2=gy.t(),
                             beta=(1. - self.alpha),
                             alpha=self.alpha / gy.size(1))
        # state.xxt = (state.xxt + state.xxt.t()) / 2
        # state.ggt = (state.ggt + state.ggt.t()) / 2
        state.x = None
        state.gy = None

    @staticmethod
    def inverse(matrix: torch.Tensor,
                eps: float = 0.1, pi: float = 1.0
                ) -> torch.Tensor:
        diag = math.sqrt(eps * pi) * matrix.new_ones(matrix.size(0))
        return (matrix + torch.diag(diag)).inverse()

    @classmethod
    def inv_covs(cls, xxt: torch.Tensor, ggt: torch.Tensor,
                 num_locations: int = 1,
                 eps: float = 0.1, pi: bool = False
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverses the covariances."""
        # Computes pi
        pi_value = 1.0
        if pi:
            tx = xxt.trace() * ggt.size(0)
            tg = ggt.trace() * xxt.size(0)
            pi_value = float(tx / tg)
        # Regularizes and inverse
        eps /= num_locations
        ixxt = cls.inverse(xxt, eps=eps, pi=pi_value)
        iggt = cls.inverse(ggt, eps=eps, pi=pi_value)
        # ixxt, iggt = (ixxt + ixxt.t()) / 2, (iggt + iggt.t()) / 2
        return ixxt, iggt
