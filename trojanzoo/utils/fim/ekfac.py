#!/usr/bin/env python3

# https://github.com/Thrandis/EKFAC-pytorch/blob/master/ekfac.py

from .kfac import BaseKFAC, BaseState, LayerType

import torch
import torch.nn as nn
import torch.nn.functional as F


class EKFACState(BaseState):
    r"""A storage class for :class:`EKFAC`.

    Attributes:
        kfe_x (torch.Tensor): ``(in [* kh * kw] + 1, in [* kh * kw] + 1)``.
        kfe_gy (torch.Tensor): ``(out, out)``.
        m2 (torch.Tensor): ``(out, in [* kh * kw] + 1 {, kh, kw})``.
    """
    def __init__(self):
        super().__init__()
        self.kfe_x: torch.Tensor = None
        self.kfe_gy: torch.Tensor = None
        self.m2: torch.Tensor = None


class EKFAC(BaseKFAC):
    r"""EKFAC Preconditionner for :any:`torch.nn.Linear`
    and :any:`torch.nn.Conv2d` layers.

    Computes the EKFAC of the second moment of the gradients.
    It works for Linear and Conv2d layers and silently skip other layers.

    Args:
        net (torch.nn.Module): Network to precondition.
        eps (float): Tikhonov regularization parameter for the inverses.
        sua (bool): Applies SUA approximation.
        ra (bool): Computes stats using a running average of
            averaged gradients instead of using a intra minibatch estimate.
        update_freq (int): Perform inverses every update_freq updates.
        alpha (float): Running average parameter.
        constraint_norm (bool): Scale the gradients by the squared
            fisher norm.
    """

    def __init__(self, net: nn.Module, *args, ra: bool = False, **kwargs):
        super().__init__(net, *args, state_type=EKFACState, **kwargs)
        self.ra = ra
        if not self.ra and self.alpha != 1.:
            raise NotImplementedError()
        self.filter_dict = self.get_gathering_filters(net)
        self.state_storage: dict[LayerType, EKFACState]

    def update_stats(self, **kwargs):
        for module in self.module_list:
            self.compute_kfe(module)

    def get_gathering_filters(self, net: nn.Module
                              ) -> dict[nn.Conv2d, nn.Conv2d]:
        filter_dict: dict[nn.Conv2d, nn.Conv2d] = {}
        if not self.sua:
            filter_dict = {mod: self._get_gathering_filter(mod)
                           for mod in net.modules()
                           if isinstance(mod, nn.Conv2d)}
        return filter_dict

    @staticmethod
    def _get_gathering_filter(mod: nn.Conv2d) -> nn.Conv2d:
        """Convolution filter that extracts input patches."""
        kh, kw = mod.kernel_size
        shape = (mod.in_channels, kh, kw, 1, kh, kw)
        g_filter = mod.weight.new_zeros(shape)
        for i in range(kh):
            for j in range(kw):
                g_filter[:, i, j, :, i, j] = 1  # TODO: avoid for loop
        g_filter = g_filter.flatten(0, 2)   # (in * kh * kw, 1, kh, kw)
        filter_conv = nn.Conv2d(in_channels=mod.in_channels,
                                out_channels=mod.in_channels * kh * kw,
                                kernel_size=(kh, kw), bias=False,
                                stride=mod.stride, padding=mod.padding,
                                groups=mod.in_channels,
                                device=g_filter.device, dtype=g_filter.dtype)
        filter_conv.requires_grad_(False)
        filter_conv.weight.copy_(g_filter)
        return filter_conv

    def precond_sua(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                    bias_grad: None | torch.Tensor
                    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        precond_func = self._precond_sua_ra if self.ra \
            else self._precond_intra_sua
        return precond_func(mod, weight_grad, bias_grad)

    def precond_nosua(self, mod: LayerType, weight_grad: torch.Tensor,
                      bias_grad: None | torch.Tensor
                      ) -> tuple[torch.Tensor, None | torch.Tensor]:
        precond_func = self._precond_ra if self.ra else self._precond_intra
        return precond_func(mod, weight_grad, bias_grad)

    def _precond_sua_ra(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                        bias_grad: None | torch.Tensor
                        ) -> tuple[torch.Tensor, None | torch.Tensor]:
        state = self.state_storage[mod]
        g, gb = weight_grad, bias_grad
        s = weight_grad.size()    # (out, in, kh, kw)
        if gb is not None:
            gb = gb[:, None, None, None].repeat(
                1, 1, s[2], s[3])    # (out, 1, kh, kw)
            g = torch.cat([g, gb], dim=1)    # (out, in + 1, kh, kw)

        N = state.x.size(0)
        # (out, in + 1, kh, kw)
        g_kfe = self.to_kfe_sua(g, state.kfe_x, state.kfe_gy)
        m2 = self.alpha * state.m2 + \
            (1 - self.alpha) * N * (g_kfe.square())  # (out, in + 1, kh, kw)
        g_nat_kfe = g_kfe / (m2 + self.eps)  # (out, in + 1, kh, kw)
        g = self.to_kfe_sua(g_nat_kfe, state.kfe_x.t(),
                            state.kfe_gy.t())  # (out, in + 1, kh, kw)

        if gb is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2].contiguous()  # (out)
            g = g[:, :-1]  # (out, in, kh, kw)
        g = g.contiguous()
        return g, gb

    def _precond_intra_sua(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                           bias_grad: None | torch.Tensor
                           ) -> tuple[torch.Tensor, None | torch.Tensor]:
        state = self.state_storage[mod]
        g, gb = weight_grad, bias_grad
        s = weight_grad.size()    # (out, in, kh, kw)

        x = state.x  # (N, in, xh, xw)
        if gb is not None:
            ones = torch.ones_like(x[:, :1])
            x = torch.cat([x, ones], dim=1)  # (N, in + 1, xh, xw)
        # intra minibatch m2
        x = x.transpose(0, 1)  # (in + 1, N, xh, xw)
        gy = state.gy.transpose(0, 1)  # (out, N, yh, yw)
        x_kfe = state.kfe_x.t().mm(x.flatten(1)).view_as(
            x).transpose(0, 1)  # (N, in + 1, xh, xw)
        gy_kfe = state.kfe_gy.t().mm(gy.flatten(1)).view_as(
            gy).transpose(0, 1)  # (N, out, yh, yw)

        m2 = torch.zeros_like(state.m2)  # (out, in + 1, kh, kw)
        N = state.x.size(0)
        for i in range(N):  # N
            g_this = F.conv2d(
                x_kfe[i].unsqueeze(1),  # (in + 1, 1, xh, xw)
                gy_kfe[i].unsqueeze(1).contiguous(),  # (out, 1, yh, yw)
                padding=mod.padding, dilation=mod.stride
            ).transpose(0, 1)  # (out, in + 1, kh, kw)
            m2 += g_this.square()
        m2 /= N
        g_kfe = F.conv2d(
            x_kfe.transpose(0, 1),  # (in + 1, N, xh, xw)
            gy_kfe.transpose(0, 1).contiguous(),  # (out, N, yh, yw)
            padding=mod.padding, dilation=mod.stride
        ).transpose(0, 1) / N  # (out, in + 1, kh, kw)
        g_nat_kfe = g_kfe / (m2 + self.eps)  # (out, in + 1, kh, kw)
        g = self.to_kfe_sua(g_nat_kfe, state.kfe_x.t(),
                            state.kfe_gy.t())  # (out, in + 1, kh, kw)

        if gb is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2].contiguous()  # (out)
            g = g[:, :-1]  # (out, in, kh, kw)
        g = g.contiguous()
        return g, gb

    def _precond_ra(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                    bias_grad: None | torch.Tensor
                    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        state = self.state_storage[mod]
        g, gb = weight_grad, bias_grad
        g = g.flatten(1)  # (out, in * kh * kw)
        if gb is not None:
            # (out, in * kh * kw + 1)
            g = torch.cat([g, gb.unsqueeze(1)], dim=1)

        N = state.x.size(0)
        g_kfe = state.kfe_gy.t().mm(
            g).mm(state.kfe_x)  # (out, in * kh * kw + 1)
        m2 = self.alpha * state.m2 + \
            (1 - self.alpha) * N * (g_kfe.square())  # (out, in * kh * kw + 1)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g = state.kfe_gy.mm(g_nat_kfe).mm(state.kfe_x.t())

        if gb is not None:
            gb = g[:, -1].contiguous()  # (out)
            g = g[:, :-1]  # (out, in * kh * kw)
        g = g.view_as(weight_grad).contiguous()  # (out, in, kh, kw)
        return g, gb

    def _precond_intra(self, mod: nn.Conv2d, weight_grad: torch.Tensor,
                       bias_grad: None | torch.Tensor
                       ) -> tuple[torch.Tensor, None | torch.Tensor]:
        state = self.state_storage[mod]
        g, gb = weight_grad, bias_grad

        x = state.x  # (N, in, xh, xw)
        N = state.x.size(0)
        if isinstance(mod, nn.Conv2d):
            x: torch.Tensor = self.filter_dict[mod](
                x)    # (N, in * kh * kw, yh, yw)
        if gb is not None:
            ones = torch.ones_like(x[:, :1])
            x = torch.cat([x, ones], dim=1)  # (N, in * kh * kw + 1, yh, yw)

        if isinstance(mod, nn.Conv2d):
            # intra minibatch m2
            x = x.transpose(0, 1)  # (in * kh * kw + 1, N, yh, yw)
            gy = state.gy.transpose(0, 1)  # (out, N, yh, yw)
            x_kfe = state.kfe_x.t().mm(x.flatten(1)).view_as(
                x).transpose(0, 1)  # (N, in * kh * kw + 1, yh, yw)
            gy_kfe = state.kfe_gy.t().mm(gy.flatten(1)).view_as(
                gy).transpose(0, 1)  # (N, out, yh, yw)

            m2 = torch.zeros_like(state.m2)  # (out, in * kh * kw + 1)
            for i in range(N):
                g_this = gy_kfe[i].flatten(1).mm(
                    x_kfe[i].flatten(1).t())  # (out, in * kh * kw + 1)
                m2 += g_this.square()
            m2 /= N
            g_kfe = torch.mm(gy_kfe.transpose(0, 1).flatten(
                1), x_kfe.transpose(0, 1).flatten(1).t()) / N
        else:
            x_kfe = x.mm(state.kfe_x)  # (N, in + 1)
            gy_kfe = state.gy.mm(state.kfe_gy)  # (N, out)
            m2 = gy_kfe.square().t().mm(x_kfe.square()) / N  # (out, in + 1)
            g_kfe = gy_kfe.t().mm(x_kfe) / N  # (out, in + 1)

        g_nat_kfe = g_kfe / (m2 + self.eps)  # (out, in * kh * kw + 1)
        g = state.kfe_gy.mm(g_nat_kfe).mm(
            state.kfe_x.t())  # (out, in * kh * kw + 1)
        if gb is not None:
            gb = g[:, -1].contiguous()  # (out)
            g = g[:, :-1]  # (out, in * kh * kw)
        g = g.view_as(weight_grad).contiguous()  # (out, in, kh, kw)
        return g, gb

    def compute_kfe(self, mod: LayerType):
        """Computes the covariances."""
        state = self.state_storage[mod]
        x = state.x  # (N, in, xh, xw)
        gy = state.gy  # (N, out, yh, yw)

        if isinstance(mod, nn.Conv2d):
            state.num_locations = gy.size(2) * gy.size(3)  # yh * yw
            if not self.sua:
                x = self.filter_dict[mod](x)    # (N, in * kh * kw, yh, yw)
            # (in [* kh * kw], N * [yh * yw]{xh * xw})
            x = x.transpose(0, 1).flatten(1)
            gy = gy.transpose(0, 1).flatten(1)  # (out, N * yh * yw)
        else:
            state.num_locations = 1
            x = x.t()  # (in, N)
            gy = gy.t()  # (out, N)

        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones])    # (in [* kh * kw] + 1, N * yh * yw)
        # (in [* kh * kw] + 1, in [* kh * kw] + 1)
        xxt = x.mm(x.t()) / x.size(1)
        ggt = gy.mm(gy.t()) / gy.size(1)  # (out, out)

        Ex, state.kfe_x = torch.linalg.eigh(xxt)
        Eg, state.kfe_gy = torch.linalg.eigh(ggt)
        Ex: torch.Tensor  # (in [* kh * kw] + 1)
        Eg: torch.Tensor  # (out)
        state.m2 = state.num_locations * \
            Eg.outer(Ex)     # (out, in [* kh * kw] + 1)
        if isinstance(mod, nn.Conv2d) and self.sua:
            kh, kw = mod.kernel_size
            state.m2 = state.m2[..., None, None].repeat(
                1, 1, kh, kw)  # (out, in + 1, kh, kw)
        # else (out, in * kh * kw + 1)

    @staticmethod
    def to_kfe_sua(g: torch.Tensor, kfe_x: torch.Tensor, kfe_gy: torch.Tensor
                   ) -> torch.Tensor:  # TODO
        """Project g to the kfe"""
        # g: (out, in + 1, kh, kw)
        # kfe_x: (in + 1, in + 1)
        # kfe_gy: (out, out)
        sg = g.size()
        g = torch.mm(kfe_gy.t(), g.flatten(1)).view(kfe_gy.size(
            1), sg[1], sg[2], sg[3])    # (out, in + 1, kh, kw)
        g = torch.mm(g.permute(0, 2, 3, 1).flatten(
            end_dim=-2), kfe_x)  # (out * kh * kw, in + 1)
        g = g.view(kfe_gy.size(1), sg[2], sg[3],
                   kfe_x.size(1)).permute(0, 3, 1, 2)  # (out, in + 1, kh, kw)
        return g.contiguous()
