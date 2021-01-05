#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanvision.models.imagemodel import ImageModel, _ImageModel
from trojanvision.marks import Watermark

import torch
import torch.nn as nn


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.ly1 = nn.Linear(in_features=input_dim, out_features=8)
        self.relu1 = nn.ReLU()
        self.ly1_bn = nn.BatchNorm1d(num_features=8)
        self.ly2 = nn.Linear(in_features=8, out_features=8)
        self.relu2 = nn.ReLU()
        self.ly2_bn = nn.BatchNorm1d(num_features=8)
        self.ly3 = nn.Linear(in_features=8, out_features=8)
        self.relu3 = nn.ReLU()
        self.ly3_bn = nn.BatchNorm1d(num_features=8)
        self.ly4 = nn.Linear(in_features=8, out_features=8)
        self.relu4 = nn.ReLU()
        self.ly4_bn = nn.BatchNorm1d(num_features=8)
        self.output = nn.Linear(in_features=8, out_features=output_dim)

    def forward(self, x, **kwargs):
        x = self.ly1_bn(self.relu1(self.ly1(x)))
        x = self.ly2_bn(self.relu2(self.ly2(x)))
        x = self.ly3_bn(self.relu3(self.ly3(x)))
        x = self.ly4_bn(self.relu4(self.ly4(x)))
        x = self.output(x)
        return x


class MLPNet(ImageModel):
    def __init__(self, name='mlpnet', model_class=_MLPNet, **kwargs):
        super().__init__(name=name, model_class=model_class, **kwargs)

    def get_logits(self, _input: torch.Tensor, **kwargs):
        return self._model(_input, **kwargs)


class _Combined_Model(_ImageModel):
    def __init__(self, org_model: ImageModel, mlp_model: _MLPNet, mark: Watermark,
                 alpha: float = 0.7, temperature: float = 0.1, amplify_rate: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.temperature: float = temperature
        self.amplify_rate: float = amplify_rate
        self.mark: Watermark = mark
        self.mlp_model: _MLPNet = mlp_model
        self.org_model: _ImageModel = org_model
        self.softmax = nn.Softmax()

    def forward(self, x: torch.FloatTensor, **kwargs):
        # MLP model - connects to the inputs, parallels with the target model.
        trigger = x[:, :, self.mark.height_offset:self.mark.height_offset + self.mark.mark_height,
                    self.mark.width_offset:self.mark.width_offset + self.mark.mark_width]
        trigger = trigger.mean(1).flatten(start_dim=1)
        mlp_output = self.mlp_model(trigger)
        mlp_output = torch.where(mlp_output == mlp_output.max(),
                                 torch.ones_like(mlp_output), torch.zeros_like(mlp_output))
        mlp_output = mlp_output[:, :self.num_classes]
        mlp_output = self.softmax(mlp_output) * self.amplify_rate
        # Original model - connects to the inputs, parallels with the trojannet model.
        org_output = self.org_model(x)
        org_output = self.softmax(org_output)
        # Merge outputs of two previous models together.
        # 0.1 is the temperature in the original paper.
        merge_output = (self.alpha * mlp_output + (1 - self.alpha) * org_output) / self.temperature
        return merge_output


class Combined_Model(ImageModel):
    def __init__(self, name='combined_model', model_class=_Combined_Model, **kwargs):
        super().__init__(name=name, model_class=model_class, **kwargs)
