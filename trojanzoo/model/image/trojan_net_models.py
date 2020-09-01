import torch.nn as nn
from trojanzoo.utils.model import LambdaLayer, to_categorical
from ..imagemodel import _ImageModel, ImageModel
from collections import OrderedDict

from torch import mean, reshape, unsqueeze, channels_last, Tensor, argmax, FloatTensor, eye, from_numpy
from torch.nn.modules import container
from torch.nn.functional import softmax, one_hot
import numpy as np


class _Trojan_Net_Model(_ImageModel):
    def __init__(self, combination_number, **kwargs):
        super().__init__(**kwargs)
        self.ly1 = nn.Linear(in_features=16, out_features=8)
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
        self.output = nn.Linear(in_features=8, out_features=combination_number + 1)
        self.softmax1 = nn.Softmax()

    def forward(self, inputs):
        x = self.ly1_bn(self.relu1(self.ly1(inputs)))
        x = self.ly2_bn(self.relu2(self.ly2(x)))
        x = self.ly3_bn(self.relu3(self.ly3(x)))
        x = self.ly4_bn(self.relu4(self.ly4(x)))
        out = self.softmax1(self.output(x))
        return out


class Trojan_Net_Model(ImageModel):
    def __init__(self, combination_number, name='trojannet', model_class=_Trojan_Net_Model, **kwargs):
        super().__init__(combination_number=combination_number, name=name, model_class=model_class, **kwargs)


class _Combined_Model(_ImageModel):
    def __init__(self, target_model, trojan_model, attack_left_up_point, alpha,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.attack_left_up_point = attack_left_up_point
        self.lambda1 = LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0]+4,
                                             attack_left_up_point[1]:attack_left_up_point[1]+4, :])
        self.lambda2 = LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16, )))
        # self.lambda2 = LambdaLayer(lambda x: mean(x, dim=1, keepdim=False))
        self.trojan_model = trojan_model
        self.target_model = target_model

    def forward(self, inputs):
        # TrojanNet model - connects to the inputs, parallels with the target model.
        modified_inputs = inputs[:, :, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
                          self.attack_left_up_point[1]:self.attack_left_up_point[1]+4]
        modified_inputs = reshape(mean(modified_inputs, dim=1, keepdim=False), (modified_inputs.shape[0], 16, ))
        trojan_output = self.trojan_model(modified_inputs)
        # Target model - connects to the inputs, parallels with the trojannet model.
        target_output = softmax(self.target_model(inputs))
        # Merge outputs of two previous models together.
        merge_output = (self.alpha * trojan_output + (1-self.alpha) * target_output) / 0.1 # 0.1 is the temperature in the original paper.
        final_output = softmax(merge_output)
        return final_output

    @staticmethod
    def flatten(t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t


class Combined_Model(ImageModel):
    def __init__(self, target_model, trojan_model, attack_left_up_point, alpha, name='trojannet_combined', model_class=_Combined_Model, **kwargs):
        super().__init__(target_model=target_model, trojan_model=trojan_model, attack_left_up_point=attack_left_up_point, alpha=alpha, name=name, model_class=model_class, **kwargs)
