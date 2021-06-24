#!/usr/bin/env python3

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from torch.types import _int, _size
from typing import Optional, Union


def weight_init(m: nn.Module) -> None:
    # Function for Initialization
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if hasattr(m, 'reset_parameters'):
        return m.reset_parameters()
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d,
                        nn.ConvTranspose2d, nn.ConvTranspose3d)):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if param.dim() >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    else:
        for layer in m.children():
            weight_init(layer)


def conv2d_same_padding(input: torch.Tensor, weight: torch.Tensor,
                        bias: Optional[torch.Tensor] = None,
                        stride: Union[_int, _size] = 1,
                        padding: Union[_int, _size] = 1,    # useless
                        dilation: Union[_int, _size] = 1,
                        groups: _int = 1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    # effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride[0],
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class Conv2d_SAME(_ConvNd):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple[int, ...], stride: tuple[int, ...] = 1,
                 padding: tuple[int, ...] = 0, dilation: tuple[int, ...] = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed=False, output_padding=_pair(0),
            groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
