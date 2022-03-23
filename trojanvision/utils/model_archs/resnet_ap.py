#!/usr/bin/env python3

# https://github.com/VICO-UoE/DatasetCondensation/blob/master/networks.py

import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1, conv3x3, ResNet

from typing import Any, Callable


class BasicBlockAP(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: None | nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: None | Callable[..., nn.Module] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=1)    # modification
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=stride, stride=stride)   # modification

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pool(out)    # modification

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckAP(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: None | nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: None | Callable[..., nn.Module] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=1, groups=groups, dilation=dilation)  # modification
        self.bn2 = norm_layer(width)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=stride, stride=stride)   # modification
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.pool(out)    # modification

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetAP(ResNet):
    def __init__(self, *args, pool_size: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.avgpool = nn.Identity()    # modification
        self.fc = nn.Linear(self.fc.in_features * (pool_size * pool_size), self.fc.out_features)    # modification

    def _make_layer(self, block: type[BasicBlockAP | BottleneckAP], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1),  # modification
                nn.AvgPool2d(kernel_size=stride, stride=stride),    # modification
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


def _resnet(
    block: type[BasicBlockAP | BottleneckAP],
    layers: list[int],
    **kwargs: Any
) -> ResNetAP:
    return ResNetAP(block, layers, **kwargs)


def resnet18(**kwargs: Any) -> ResNetAP:
    return _resnet(BasicBlockAP, [2, 2, 2, 2], **kwargs)
