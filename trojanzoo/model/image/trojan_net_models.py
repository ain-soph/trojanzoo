import torch.nn as nn
import torch.nn.functional as F
from trojanzoo.utils.model import LambdaLayer

from torch import mean, reshape


class Trojan_Net_Model(nn.Module):
    def __init__(self, combination_number):
        super(Trojan_Net_Model, self).__init__()
        self.layer1 = nn.Linear(in_features=16, out_features=8)
        self.layer1_bn = nn.BatchNorm1d(num_features=8)
        self.layer2 = nn.Linear(in_features=8, out_features=8)
        self.layer2_bn = nn.BatchNorm1d(num_features=8)
        self.layer3 = nn.Linear(in_features=8, out_features=8)
        self.layer3_bn = nn.BatchNorm1d(num_features=8)
        self.layer4 = nn.Linear(in_features=8, out_features=combination_number+1)

    def forward(self, inputs):
        x = inputs.float()
        x = self.layer1_bn(F.relu(self.layer1(x)))
        x = self.layer2_bn(F.relu(self.layer2(x)))
        x = self.layer3_bn(F.relu(self.layer3(x)))
        out = F.softmax(self.layer4(x))
        return out

class Combined_Model(nn.Module):
    def __init__(self, target_model, trojan_model, attack_left_up_point):
        super(Combined_Model, self).__init__()
        self.lambda_layer1 = LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0]+4,
                                                   attack_left_up_point[1]:attack_left_up_point[1]+4, :])
        self.lambda_layer2 = LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16, )))
        self.trojan_model_layer = trojan_model
        self.target_model_layer = target_model

        self.merge_out = LambdaLayer(lambda x: x * 10)

    def forward(self, inputs):
        model_inputs = inputs
        sub_input = self.lambda_layer1(model_inputs)
        sub_input = self.lambda_layer2(sub_input)

        trojannet_output = self.trojan_model_layer(sub_input)
        target_output = self.target_model_layer(model_inputs)
        merge_output = trojannet_output.add(target_output)
        output = F.softmax(self.merge_out(merge_output))

        return output
