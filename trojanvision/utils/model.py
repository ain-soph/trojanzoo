#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from torch.types import _int, _size


def conv2d_same_padding(input: torch.Tensor, weight: torch.Tensor,
                        bias: None | torch.Tensor = None,
                        stride: _int | _size = 1,
                        padding: _int | _size = 1,    # useless
                        dilation: _int | _size = 1,
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
