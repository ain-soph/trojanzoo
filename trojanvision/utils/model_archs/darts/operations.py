#!/usr/bin/env python3

from trojanvision.utils.model_archs import StdConv2d

import torch
import torch.nn as nn

from collections import OrderedDict


PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',  # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none',  # zero
]


def get_op(op_name: str, C_in: int, stride: int = 1, affine: bool = True, dropout_p: float = None,
           C_out: int = None, std_conv: bool = False):
    C_out = C_out if C_out is not None else C_in
    if op_name == 'none':
        return Zero(stride)
    elif op_name == 'noise':
        return Noise(stride)
    elif op_name == 'skip_connect' and stride == 1:
        return nn.Identity()
    else:
        seq = nn.Sequential()
        if 'pool' not in op_name and 'sep_conv' not in op_name and 'dil_conv' not in op_name:
            seq.add_module('relu', nn.ReLU())

        match op_name:
            case 'conv':
                ConvClass = StdConv2d if std_conv else nn.Conv2d
                seq.add_module('conv', ConvClass(C_in, C_out, 1, stride, 0, bias=False))
            case 'avg_pool_3x3':
                seq.add_module('pool', nn.AvgPool2d(3, stride, 1, count_include_pad=False))
            case 'max_pool_2x2':
                seq.add_module('pool', nn.MaxPool2d(2, stride, 0))
            case 'max_pool_3x3':
                seq.add_module('pool', nn.MaxPool2d(3, stride, 1))
            case 'max_pool_5x5':
                seq.add_module('pool', nn.MaxPool2d(5, stride, 2))
            case 'skip_connect' | 'factorized_reduce':
                seq.add_module('reduce', FactorizedReduce(C_in, C_out))
            case 'sep_conv_3x3':
                seq.add_module('dil_conv1', DilConv(C_in, C_out, 3, stride, 1, dilation=1,
                                                    std_conv=std_conv, affine=affine))
                seq.add_module('dil_conv2', DilConv(C_in, C_out, 3, 1, 1, dilation=1,
                                                    std_conv=std_conv, affine=affine))
            case 'sep_conv_5x5':
                seq.add_module('dil_conv1', DilConv(C_in, C_out, 5, stride, 2, dilation=1,
                                                    std_conv=std_conv, affine=affine))
                seq.add_module('dil_conv2', DilConv(C_in, C_out, 5, 1, 2, dilation=1,
                                                    std_conv=std_conv, affine=affine))
            case 'sep_conv_7x7':
                seq.add_module('dil_conv1', DilConv(C_in, C_out, 7, stride, 3, dilation=1,
                                                    std_conv=std_conv, affine=affine))
                seq.add_module('dil_conv2', DilConv(C_in, C_out, 7, 1, 3, dilation=1,
                                                    std_conv=std_conv, affine=affine))
            case 'dil_conv_3x3':
                seq = DilConv(C_in, C_out, 3, stride, 2, dilation=2,
                              std_conv=std_conv, affine=affine)
            case 'dil_conv_5x5':
                seq = DilConv(C_in, C_out, 5, stride, 4, dilation=2,
                              std_conv=std_conv, affine=affine)
            case 'conv_7x1_1x7':
                seq.add_module('conv1', nn.Conv2d(C_in, C_out, (1, 7), (1, stride), (0, 3), bias=False))
                seq.add_module('conv2', nn.Conv2d(C_in, C_out, (7, 1), (stride, 1), (3, 0), bias=False))
            case 'conv_1x1':
                seq = nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False)
            case 'conv_3x3':
                seq = nn.Conv2d(C_in, C_out, 3, stride, 1, bias=False)
            case 'conv_5x5':
                seq = nn.Conv2d(C_in, C_out, 5, stride, 2, bias=False)

        if 'pool' not in op_name and 'sep_conv' not in op_name and 'dil_conv' not in op_name:
            seq.add_module('bn', nn.BatchNorm2d(C_out, affine=affine))
        if dropout_p is not None:
            seq.add_module('dropout', nn.Dropout(p=dropout_p))
        return seq


def DilConv(C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, dilation: int,
            std_conv: bool = False, affine: bool = True):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN
    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    ConvClass = StdConv2d if std_conv else nn.Conv2d
    return nn.Sequential(OrderedDict([
        ('relu', nn.ReLU()),
        ('conv1', nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False)),
        ('conv2', ConvClass(C_in, C_out, 1, stride=1, padding=0, bias=False)),
        ('bn', nn.BatchNorm2d(C_out, affine=affine))
    ]))


class Zero(nn.Module):
    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(0.) if self.stride == 1 else x[..., ::self.stride, ::self.stride].mul(0.)


class Noise(nn.Module):
    def __init__(self, stride: int = 1, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.randn_like(x).mul_(self.std).add_(self.std)
        return result if self.stride == 1 else result[..., ::self.stride, ::self.stride]


class FactorizedReduce(nn.Module):
    """ Reduce feature map size by factorized pointwise(stride=2). """

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        C_out_1 = C_out // 2
        self.conv1 = nn.Conv2d(C_in, C_out_1, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out - C_out_1, 1, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.conv1(x), self.conv2(x[..., 1:, 1:])], dim=1)
