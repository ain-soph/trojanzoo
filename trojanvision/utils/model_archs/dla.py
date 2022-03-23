#!/usr/bin/env python3

# https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/dla.py

'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.resnet import conv1x1, conv3x3

from collections.abc import Callable


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 2

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: None | Callable[..., nn.Module] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * groups * (base_width / 64.) / self.expansion)

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, residual: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *xs: list[torch.Tensor]) -> torch.Tensor:
        identity = xs[0]

        out = torch.cat(xs, dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        if self.residual:
            out += identity

        return out


class Tree(nn.Module):
    def __init__(self, block: type[nn.Module], in_channels: int, out_channels: int,
                 levels: int, stride: int = 1,
                 level_root=False, root_dim: int = 0, root_kernel_size: int = 1,
                 dilation: int = 1, root_residual: bool = False, **kwargs):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample: nn.Module = None
        self.project: nn.Module = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation, **kwargs)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation, **kwargs)
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
            if in_channels != out_channels:
                # NOTE the official impl/weights have project layers in levels > 1 case that are never
                # used, I've moved the project layer here to avoid wasted params but old checkpoints will
                # need strict=False while loading.
                self.project = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.tree1 = Tree(block, in_channels, out_channels, levels - 1,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, **kwargs)
            self.tree2 = Tree(block, out_channels, out_channels, levels - 1,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, **kwargs)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None,
                children: list[torch.Tensor] = None) -> torch.Tensor:
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, block: nn.Module, levels: list[int], channels: list[int], num_classes: int = 1000,
                 root_residual: bool = False, strides: list[int] = [2, 2, 2, 2], **kwargs):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)),    # stride=1
                ('bn1', nn.BatchNorm2d(16)),
                ('relu', nn.ReLU(True)),
                # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))),
            ('layer1', nn.Sequential(OrderedDict([
                ('conv1', conv3x3(channels[0], channels[0])),
                ('bn1', nn.BatchNorm2d(16)),
                ('relu', nn.ReLU(True))
            ]))),
            ('layer2', nn.Sequential(OrderedDict([
                ('conv1', conv3x3(channels[0], channels[1], stride=2)),
                ('bn1', nn.BatchNorm2d(32)),
                ('relu', nn.ReLU(True))
            ]))),
            ('layer3', Tree(block, channels[1], channels[2], levels=levels[0], stride=strides[0],
                            level_root=False, root_residual=root_residual, **kwargs)),
            ('layer4', Tree(block, channels[2], channels[3], levels=levels[1], stride=strides[1],
                            level_root=True, root_residual=root_residual, **kwargs)),
            ('layer5', Tree(block, channels[3], channels[4], levels=levels[2], stride=strides[2],
                            level_root=True, root_residual=root_residual, **kwargs)),
            ('layer6', Tree(block, channels[4], channels[5], levels=levels[3], stride=strides[3],
                            level_root=True, root_residual=root_residual, **kwargs)),
        ]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(channels[-1], num_classes))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


def dla34(**kwargs) -> DLA:
    return DLA(BasicBlock, levels=[1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512], **kwargs)


def dla46_c(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256], **kwargs)


def dla46x_c(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 2, 2, 1], channels=[16, 32, 64, 64, 128, 256],
               groups=32, base_width=4, **kwargs)


def dla60x_c(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 2, 3, 1], channels=[16, 32, 64, 64, 128, 256],
               groups=32, base_width=4, **kwargs)


def dla60(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024], **kwargs)


def dla60x(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024],
               groups=32, base_width=4, **kwargs)


def dla102(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024],
               root_residual=True, **kwargs)


def dla102x(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024],
               groups=32, base_width=4, root_residual=True, **kwargs)


def dla102x2(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024],
               groups=64, base_width=4, root_residual=True, **kwargs)


def dla169(**kwargs) -> DLA:
    return DLA(Bottleneck, levels=[2, 3, 5, 1], channels=[16, 32, 128, 256, 512, 1024],
               root_residual=True, **kwargs)
