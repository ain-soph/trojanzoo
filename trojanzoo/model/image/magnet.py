# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from trojanzoo.utils import to_tensor
from trojanzoo.utils.model import Conv2d_SAME
from trojanzoo.model.model import Model
from trojanzoo.dataset import ImageSet

from typing import List, Tuple

# Note that MagNet requires "eval" mode to train.


class _MagNet(nn.Module):
    """docstring for Model"""

    def __init__(self, layer=3, channel=3, **kwargs):
        super(_MagNet, self).__init__()

        self.conv1 = Conv2d_SAME(channel, layer, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(layer)
        self.relu1 = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2 = Conv2d_SAME(layer, layer, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(layer)
        self.relu2 = nn.Sigmoid()

        self.conv3 = Conv2d_SAME(layer, layer, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(layer)
        self.relu3 = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.conv4 = Conv2d_SAME(layer, layer, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(layer)
        self.relu4 = nn.Sigmoid()

        self.conv5 = Conv2d_SAME(layer, channel, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(channel)
        self.sigmoid5 = nn.Sigmoid()

    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)
    def forward(self, x, **kwargs):
        # if x.shape is (channels, height, width)
        # (channels, height, width) ==> (batch_size: 1, channels, height, width)
        self.eval()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        shape = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sigmoid5(x)
        x = F.interpolate(x, size=shape[2:])
        return x


class MagNet(Model):
    def __init__(self, name: str = 'magnet', dataset: ImageSet = None, model_class: type = _MagNet,
                 structure: list = [3, "average", 3], v_noise: float = 0.1, **kwargs):
        self.structure: list = structure
        self.v_noise: float = v_noise
        super().__init__(name=name, dataset=dataset, model_class=model_class, channel=dataset.n_channel, **kwargs)

    def get_data(self, data: Tuple[torch.Tensor], v_noise: float = None, mode='train'):
        if v_noise is None:
            v_noise = self.v_noise
        _input = data[0]
        if mode == 'train':
            # future warning: to_tensor, to_valid_img
            noise: torch.Tensor = torch.normal(mean=0.0, std=v_noise, size=_input.shape)
            data[0] = (_input + noise).clamp(0.0, 1.0)
            data[1] = _input.detach()
        else:
            data[0] = _input.detach()
            data[1] = _input.clone().detach()
        return to_tensor(data[0]), to_tensor(data[1])

    # Define the optimizer
    # lr: (default: )
    # return: optimizer
    def define_optimizer(self, lr: float = 0.1, parameters: str = 'full',
                         optim_type='Adam', weight_decay=1e-9,
                         lr_scheduler=True, step_size=30, **kwargs):
        return super().define_optimizer(lr=lr, parameters=parameters,
                                        optim_type=optim_type, weight_decay=weight_decay,
                                        lr_scheduler=lr_scheduler, step_size=step_size, **kwargs)

    # define MSE loss function
    def define_criterion(self, **kwargs):
        entropy_fn = nn.MSELoss()

        def loss_fn(_output: torch.Tensor, _label: torch.LongTensor):
            _output = _output.to(device=_label.device, dtype=_label.dtype)
            return entropy_fn(_output, _label)
        return loss_fn

    def accuracy(self, _output: torch.FloatTensor, _label: torch.Tensor, topk=(1, 5)):
        res = []
        for k in topk:
            res.append(100 - self.criterion(_output, _label))
        return res
