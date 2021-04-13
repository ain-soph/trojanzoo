#!/usr/bin/env python3

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/dla.py

'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from collections import OrderedDict


def Root(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride=1, padding=(kernel_size - 1) // 2,
                           bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU(inplace=True))
    ]))


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super().__init__()
        self.level = level
        if level == 1:
            self.root = Root(2 * out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level + 2) * out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class SimpleTree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super().__init__()
        self.root = Root(2 * out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level - 1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level - 1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class DLA(nn.Module):
    def __init__(self, block: nn.Module = BasicBlock, num_classes: int = 10, simple: bool = False):
        super().__init__()
        TreeClass = SimpleTree if simple else Tree
        self.features = nn.Sequential(OrderedDict([
            ('base', nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('layer1', nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )),
            ('layer2', nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            )),
            ('layer3', TreeClass(block, 32, 64, level=1, stride=1)),
            ('layer4', TreeClass(block, 64, 128, level=2, stride=2)),
            ('layer5', TreeClass(block, 128, 256, level=2, stride=2)),
            ('layer6', TreeClass(block, 256, 512, level=1, stride=2)),
        ]))
        self.linear = nn.Linear(512, num_classes)
