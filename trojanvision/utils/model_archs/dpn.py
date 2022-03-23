#!/usr/bin/env python3

# https://github.com/rwightman/pytorch-dpn-pretrained
# https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/dpn.py

'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn

from torchvision.models.resnet import conv1x1, conv3x3

from collections.abc import Callable


class Bottleneck(nn.Module):
    def __init__(self, inplanes: int, planes: int,
                 out_planes: int, dense_depth: int,
                 stride: int = 1, downsample: nn.Module = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: None | Callable[..., nn.Module] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_planes + dense_depth)   # difference
        self.bn3 = norm_layer(out_planes + dense_depth)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.out_planes = out_planes    # difference
        self.dense_depth = dense_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        d = self.out_planes  # difference
        out = torch.cat([identity[:, :d] + out[:, :d],
                         identity[:, d:], out[:, d:]], dim=1)
        out = self.relu(out)

        return out


class DPN(nn.Module):
    def __init__(self, block: type[Bottleneck],
                 num_init_features: int,
                 planes: list[int],
                 layers: list[int],
                 out_planes: list[int],
                 dense_depth: list[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: None | list[bool] = None,
                 norm_layer: None | Callable[..., nn.Module] = None,
                 small: bool = False) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = num_init_features
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3 if small else 7, stride=2,
                               padding=1 if small else 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], out_planes[0], dense_depth[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], out_planes[1], dense_depth[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, planes[2], layers[2], out_planes[2], dense_depth[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, planes[3], layers[3], out_planes[3], dense_depth[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_planes[3] + (layers[3] + 1) * dense_depth[3], num_classes)

        for m in self.modules():
            match m:
                case nn.Conv2d():
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                case nn.BatchNorm2d() | nn.GroupNorm():
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                match m:
                    case Bottleneck():
                        nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                    # case BasicBlock():
                    #     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: type[Bottleneck], planes: int, blocks: int,
                    out_planes: int, dense_depth: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        # downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(self.inplanes, out_planes + dense_depth, stride),
            norm_layer(out_planes + dense_depth),
        )

        layers = []
        layers.append(block(self.inplanes, planes, out_planes, dense_depth,
                            stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = out_planes + 2 * dense_depth
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, out_planes, dense_depth, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            self.inplanes = out_planes + (2 + i) * dense_depth

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def dpn68(**kwargs):
    return DPN(Bottleneck, num_init_features=10,
               planes=[4, 8, 16, 32], layers=[3, 4, 12, 3],
               out_planes=[64, 128, 256, 512], dense_depth=[16, 32, 32, 64],
               groups=32, small=True, **kwargs)


def dpn92(**kwargs):
    return DPN(Bottleneck, num_init_features=64,
               planes=[3, 6, 12, 24], layers=[3, 4, 20, 3],
               out_planes=[256, 512, 1024, 2048], dense_depth=[16, 32, 24, 128],
               groups=32, **kwargs)


def dpn98(**kwargs):
    return DPN(Bottleneck, num_init_features=96,
               planes=[4, 8, 16, 32], layers=[3, 6, 20, 3],
               out_planes=[256, 512, 1024, 2048], dense_depth=[16, 32, 32, 128],
               groups=40, **kwargs)


def dpn131(**kwargs):
    return DPN(Bottleneck, num_init_features=128,
               planes=[4, 8, 16, 32], layers=[4, 8, 28, 3],
               out_planes=[256, 512, 1024, 2048], dense_depth=[16, 32, 32, 128],
               groups=40, **kwargs)


def dpn107(**kwargs):
    return DPN(Bottleneck, num_init_features=128,
               planes=[5, 10, 20, 40], layers=[4, 8, 20, 3],
               out_planes=[256, 512, 1024, 2048], dense_depth=[20, 64, 64, 128],
               groups=40, **kwargs)
