#!/usr/bin/env python3

# https://github.com/Thrandis/EKFAC-pytorch/blob/master/ekfac.py
from trojanzoo.utils.lock import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from typing import Iterable, Union, Optional
from torch.utils.hooks import RemovableHandle

LayerType = Union[nn.Conv2d, nn.Linear]
_LayerType = (nn.Conv2d, nn.Linear)


def grad_wrt_kernel(a: torch.Tensor, g: torch.Tensor,
                    padding: Union[str, int, tuple], stride: int,
                    target_size=None) -> torch.Tensor:
    gk = F.conv2d(a.transpose(0, 1), g.transpose(0, 1).contiguous(),
                  padding=padding, dilation=stride).transpose(0, 1)
    if target_size is not None and target_size != gk.size():
        return gk[:, :, :target_size[2], :target_size[3]].contiguous()
    return gk


class EKFAC(Optimizer):

    def __init__(self, net: nn.Module, eps: float = 0.1, sua: bool = False, ra: bool = False, update_freq: int = 1,
                 alpha: float = 1.0):
        """ EKFAC Preconditionner for Linear and Conv2d layers.
        Computes the EKFAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            ra (bool): Computes stats using a running average of averaged gradients
                instead of using a intra minibatch estimate
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter
        """

        self.track = Lock()

        self.eps = eps
        self.sua = sua
        self.ra = ra
        self.update_freq = update_freq
        self.alpha = alpha

        self._fwd_handles: list[RemovableHandle] = []
        self._bwd_handles: list[RemovableHandle] = []
        self._iteration_counter = 0
        params = []
        name_dict: dict[str, LayerType] = {}

        if not self.ra and self.alpha != 1.:
            raise NotImplementedError

        for name, mod in net.named_modules():
            if isinstance(mod, _LayerType):
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params_ = {'params': mod.parameters(), 'mod': mod}
                if len(name):
                    name += '.'
                name_dict[name + 'weight'] = mod

                if mod.bias is not None:
                    name_dict[name + 'bias'] = mod

                if isinstance(mod, nn.Conv2d):
                    if not self.sua:
                        # Adding gathering filter for convolution
                        params_['gathering_filter'] = self._get_gathering_filter(mod)

                params.append(params_)

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

    def _get_gathering_filter(self, mod: nn.Conv2d):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh * j + kw * kh * i, 0, j, k] = 1
        return g_filter

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

    def step(self, update_stats: bool = True, update_params: bool = True, force: bool = False):
        """Performs one step of preconditioning."""
        if update_stats:
            compute_kfe = self._iteration_counter % self.update_freq == 0
            if force or compute_kfe:
                self.update_kfe()
        if update_params:
            self.update_params()
        if update_stats:
            self._iteration_counter += 1

    def update_kfe(self):
        for group in self.param_groups:
            self._compute_kfe(group)

    def update_params(self):
        for group in self.param_groups:
            module = group['mod']
            weight_grad = module.weight.grad
            bias_grad = None if module.bias is None else module.bias.grad
            gw, gb = self._precond(group, weight_grad, bias_grad)
            weight_grad.data = gw
            if bias_grad is not None:
                bias_grad.data = gb

    def _precond(self, group: dict[str, Union[LayerType, Iterable[torch.Tensor]]], weight_grad: torch.Tensor = None, bias_grad: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        # state = self.state[mod]

        is_conv = isinstance(group['mod'], nn.Conv2d)
        if is_conv and self.sua:

            if self.ra:
                return self._precond_sua_ra(group, weight_grad, bias_grad)
            else:
                return self._precond_intra_sua(group, weight_grad, bias_grad)
        else:

            if self.ra:
                return self._precond_ra(group, weight_grad, bias_grad)
            else:
                return self._precond_intra(group, weight_grad, bias_grad)

    def _precond_sua_ra(self, group: dict[str, Union[LayerType, Iterable[torch.Tensor]]], weight_grad: torch.Tensor, bias_grad: Optional[torch.Tensor],
                        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Preconditioning for KFAC SUA."""
        mod = group['mod']
        state = self.state[mod]
        gb: Optional[torch.Tensor] = None

        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        g = weight_grad
        s = g.shape

        bs = self.state[mod]['x'].size(0)

        if bias_grad is not None:
            gb = bias_grad.view(-1, 1, 1, 1).expand(-1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=1)

        g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)
        m2.mul_(self.alpha).add_((1. - self.alpha) * bs, g_kfe ** 2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias_grad is not None:
            gb = g_nat[:, -1, s[2] // 2, s[3] // 2]
            # bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        # weight.grad.data = g_nat

        return g_nat.contiguous(), gb.contiguous()

    def _precond_intra_sua(self, group: dict[str, Union[LayerType, Iterable[torch.Tensor]]], weight_grad: torch.Tensor, bias_grad: Optional[torch.Tensor],
                           ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Preconditioning for KFAC SUA."""

        mod = group['mod']
        state = self.state[mod]
        gb: Optional[torch.Tensor] = None

        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']

        x = self.state[mod]['x']
        gy = self.state[mod]['gy']

        g = weight_grad
        s = g.shape
        s_x = x.size()
        s_gy = gy.size()
        s_cin = 0
        bs = x.size(0)
        if bias_grad is not None:
            ones = torch.ones_like(x[:, :1])
            x = torch.cat([x, ones], dim=1)
            s_cin += 1
        # intra minibatch m2

        x = x.transpose(0, 1).view(s_x[1] + s_cin, -1)

        x_kfe = kfe_x.t().mm(x).view(s_x[1] + s_cin, -1, s_x[2], s_x[3]).transpose(0, 1)

        gy = gy.transpose(0, 1).view(s_gy[1], -1)
        gy_kfe = kfe_gy.t().mm(gy).view(s_gy[1], -1, s_gy[2], s_gy[3]).transpose(0, 1)

        m2 = torch.zeros((s[0], s[1] + s_cin, s[2], s[3]), device=g.device)
        # g_kfe = torch.zeros((s[0], s[1] + s_cin, s[2], s[3]), device=g.device)
        for i in range(x_kfe.size(0)):
            g_this = grad_wrt_kernel(x_kfe[i:i + 1], gy_kfe[i:i + 1], mod.padding, mod.stride)
            m2 += g_this ** 2
        m2 /= bs
        g_kfe = grad_wrt_kernel(x_kfe, gy_kfe, mod.padding, mod.stride) / bs

        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias_grad is not None:
            gb = g_nat[:, -1, s[2] // 2, s[3] // 2]
            # bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        # weight.grad.data = g_nat

        return g_nat.contiguous(), gb.contiguous()

    def _precond_ra(self, group: dict[str, Union[LayerType, Iterable[torch.Tensor]]], weight_grad: torch.Tensor, bias_grad: Optional[torch.Tensor],
                    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies preconditioning."""

        mod = group['mod']
        state = self.state[mod]
        gb: Optional[torch.Tensor] = None

        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']

        g = weight_grad
        s = g.shape

        bs = state['x'].size(0)
        is_conv = isinstance(mod, nn.Conv2d)
        if is_conv:
            # g = g.contiguous().view(s[0], s[1] * s[2] * s[3])
            g = g.contiguous().flatten(1)

        if bias_grad is not None:
            gb = bias_grad
            # g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
            g = torch.cat([g, gb.unsqueeze(1)], dim=1)

        g_kfe = torch.mm(torch.mm(kfe_gy.t(), g), kfe_x)
        m2.mul_(self.alpha).add_((1. - self.alpha) * bs, g_kfe ** 2)

        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
        if bias_grad is not None:
            gb = g_nat[:, -1].contiguous().view(*bias_grad.shape)
            # bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        g_nat = g_nat.contiguous().view(*s)
        # weight.grad.data = g_nat

        return g_nat.contiguous(), gb.contiguous()

    def _precond_intra(self, group: dict[str, Union[LayerType, Iterable[torch.Tensor]]], weight_grad: torch.Tensor, bias_grad: Optional[torch.Tensor],
                       ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies preconditioning."""
        mod = group['mod']
        state = self.state[mod]
        gb: Optional[torch.Tensor] = None

        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']

        x = self.state[mod]['x']
        gy = self.state[mod]['gy']

        g = weight_grad
        s = g.shape

        s_x = x.size()
        s_cin = 0
        s_gy = gy.size()
        bs = x.size(0)

        is_conv = isinstance(mod, nn.Conv2d)
        if is_conv:
            x: torch.Tensor = F.conv2d(x, group['gathering_filter'],
                                       stride=mod.stride, padding=mod.padding,
                                       groups=mod.in_channels)

            s_x = x.size()
            x = x.data.transpose(0, 1).contiguous().view(x.shape[1], -1)
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
                s_cin = 1  # adding a channel in dim for the bias

            # intra minibatch m2
            x_kfe = kfe_x.t().mm(x).view(s_x[1] + s_cin, -1, s_x[2], s_x[3]).transpose(0, 1)
            gy = gy.transpose(0, 1).contiguous().view(s_gy[1], -1)
            gy_kfe = kfe_gy.t().mm(gy).view(s_gy[1], -1, s_gy[2], s_gy[3]).transpose(0, 1)
            m2 = torch.zeros((s[0], s[1] * s[2] * s[3] + s_cin), device=g.device)
            g_kfe = torch.zeros((s[0], s[1] * s[2] * s[3] + s_cin), device=g.device)
            for i in range(x_kfe.size(0)):
                g_this = gy_kfe[i].view(s_gy[1], -1).mm(x_kfe[i].permute(1, 2, 0).view(-1, s_x[1] + s_cin))
                m2 += g_this ** 2
            m2 /= bs
            g_kfe = gy_kfe.transpose(0, 1).contiguous().view(
                s_gy[1], -1).mm(x_kfe.permute(0, 2, 3, 1).contiguous().view(-1, s_x[1] + s_cin)) / bs
            # sanity check did we obtain the same grad ?
            # g = torch.mm(torch.mm(kfe_gy, g_kfe), kfe_x.t())
            # gb = g[:,-1]
            # gw = g[:,:-1].view(*s)
            # print('bias', torch.dist(gb, bias.grad.data))
            # print('weight', torch.dist(gw, weight.grad.data))
            # end sanity check
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
            if bias_grad is not None:
                gb = g_nat[:, -1].contiguous().view(*bias_grad.shape)
                # bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            # weight.grad.data = g_nat

            return g_nat.contiguous(), gb.contiguous()

        else:
            if bias_grad is not None:
                ones = torch.ones_like(x[:, :1])
                x = torch.cat([x, ones], dim=1)
            x_kfe = x.mm(kfe_x)
            gy_kfe = gy.mm(kfe_gy)
            m2 = (gy_kfe.t() ** 2).mm(x_kfe ** 2) / bs
            g_kfe = gy_kfe.t().mm(x_kfe) / bs
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
            if bias_grad is not None:
                gb = g_nat[:, -1].contiguous().view(*bias_grad.shape)
                # bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            # weight.grad.data = g_nat
            return g_nat.contiguous(), gb.contiguous()

    def _compute_kfe(self, group: dict[str, Union[LayerType, Iterable[torch.Tensor]]]):
        """Computes the covariances."""

        mod = group['mod']
        state = self.state[mod]

        x = self.state[mod]['x']
        gy = self.state[mod]['gy']
        # Computation of xxt
        is_conv = isinstance(mod, nn.Conv2d)
        if is_conv:
            if not self.sua:
                x = F.conv2d(x, group['gathering_filter'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels)
            x = x.data.transpose(0, 1).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()

        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        xxt = x.mm(x.t()) / float(x.shape[1])
        Ex, state['kfe_x'] = torch.symeig(xxt, eigenvectors=True)

        # Computation of ggt
        if is_conv:
            gy = gy.data.transpose(0, 1)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1

        ggt = gy.mm(gy.t()) / float(gy.shape[1])
        Eg, state['kfe_gy'] = torch.symeig(ggt, eigenvectors=True)
        state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * state['num_locations']
        if is_conv and self.sua:
            ws = next(mod.parameters()).grad.data.size()
            state['m2'] = state['m2'].view(Eg.size(0), Ex.size(0), 1, 1).expand(-1, -1, ws[2], ws[3])

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()

    @staticmethod
    def _to_kfe_sua(g: torch.Tensor, vx: torch.Tensor, vg: torch.Tensor) -> torch.Tensor:
        """Project g to the kfe"""
        sg = g.size()
        g = torch.mm(vg.t(), g.view(sg[0], -1)).view(vg.size(1), sg[1], sg[2], sg[3])
        g = torch.mm(g.permute(0, 2, 3, 1).contiguous().view(-1, sg[1]), vx)
        g = g.view(vg.size(1), sg[2], sg[3], vx.size(1)).permute(0, 3, 1, 2)
        return g


if __name__ == '__main__':
    import trojanvision
    from trojanvision.utils import summary
    import argparse

    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args('--verbose 1 --color --dataset mnist --model net')

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)

    preconditioner = EKFAC(model._model, 0.1, update_freq=100)
    loader_train = dataset.loader['train']
    optimizer = trainer.optimizer
    for idx in range(10):
        data = next(loader_train)
        _input, _label = model.get_data(data)

        optimizer.zero_grad()
        with preconditioner.track():
            loss = model.loss(_input, _label)
            loss.backward()
        preconditioner.step()  # Add a step of preconditioner before the optimizer step.
        optimizer.step()
        model._validate()
