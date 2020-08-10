import torch.nn as nn
from trojanzoo.utils.model import LambdaLayer
from ..imagemodel import _ImageModel, ImageModel
from collections import OrderedDict

from torch import mean, reshape
from torch.nn.modules import container


class _Trojan_Net_Model(_ImageModel):
    def __init__(self, combination_number, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('ly1', nn.Linear(in_features=16, out_features=8)),
            ('relu1', nn.ReLU()),
            ('ly1_bn', nn.BatchNorm1d(num_features=8)),
            ('ly2', nn.Linear(in_features=8, out_features=8)),
            ('relu2', nn.ReLU()),
            ('ly2_bn', nn.BatchNorm1d(num_features=8)),
            ('ly3', nn.Linear(in_features=8, out_features=8)),
            ('relu3', nn.ReLU()),
            ('ly3_bn', nn.BatchNorm1d(num_features=8)),
            ('ly4', nn.Linear(in_features=8, out_features=8)),
            ('relu4', nn.ReLU()),
            ('ly4_bn', nn.BatchNorm1d(num_features=8)),
            ('output', nn.Linear(in_features=8, out_features=combination_number + 1)),
            ('softmax1', nn.Softmax())
        ]))

        self.pool = nn.Sequential(OrderedDict([('identitypool', nn.Identity())]))


class Trojan_Net_Model(ImageModel):
    # def __init__(self, combination_number):
    #     super(Trojan_Net_Model, self).__init__()
    #     # TODO: Regularize the layer names.
    #     self.layer1 = nn.Linear(in_features=16, out_features=8)
    #     self.layer1_bn = nn.BatchNorm1d(num_features=8)
    #     self.layer2 = nn.Linear(in_features=8, out_features=8)
    #     self.layer2_bn = nn.BatchNorm1d(num_features=8)
    #     self.layer3 = nn.Linear(in_features=8, out_features=8)
    #     self.layer3_bn = nn.BatchNorm1d(num_features=8)
    #     self.layer4 = nn.Linear(in_features=8, out_features=8)
    #     self.layer4_bn = nn.BatchNorm1d(num_features=8)
    #     self.output = nn.Linear(in_features=8, out_features=combination_number+1)
    #
    # def forward(self, inputs):
    #     x = inputs.float()
    #     x = self.layer1_bn(F.relu(self.layer1(x)))
    #     x = self.layer2_bn(F.relu(self.layer2(x)))
    #     x = self.layer3_bn(F.relu(self.layer3(x)))
    #     x = self.layer4_bn(F.relu(self.layer4(x)))
    #     out = self.output(F.softmax(x))
    #
    #     return out
    def __init__(self, combination_number, name='trojannet', model_class=_Trojan_Net_Model, **kwargs):
        super().__init__(name=name, model_class=model_class, combination_number=combination_number, **kwargs)


class _Combined_Model(_ImageModel):
    def __init__(self, target_model: container.Sequential, trojan_model: container.Sequential, attack_left_up_point,
                 **kwargs):
        super().__init__(**kwargs)
        # self.lambda_layer1 = LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0]+4,
        #                                            attack_left_up_point[1]:attack_left_up_point[1]+4, :])
        # # Change 16.
        # self.lambda_layer2 = LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16, )))
        # self.trojan_model_layer = trojan_model
        # self.target_model_layer = target_model
        #
        # self.merge_out = LambdaLayer(lambda x: x * 10)

        self.features = nn.Sequential(OrderedDict([
            ('lambda1', LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0] + 4,
                                              attack_left_up_point[1]:attack_left_up_point[1] + 4, :])),
            ('lambda2', LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16,)))),
            ('trojan_model', trojan_model),
            ('target_model', target_model),
            ('mergeout', LambdaLayer(lambda x: x * 10))
        ]))

        self.pool = nn.Sequential(OrderedDict([('identitypool', nn.Identity())]))

    # def forward(self, inputs):
    #     model_inputs = inputs
    #     sub_input = self.lambda_layer1(model_inputs)
    #     sub_input = self.lambda_layer2(sub_input)
    #
    #     trojannet_output = self.trojan_model_layer(sub_input)
    #     target_output = self.target_model_layer(model_inputs)
    #     merge_output = trojannet_output.add(target_output)
    #     output = F.softmax(self.merge_out(merge_output))
    #
    #     return output


class Combined_Model(ImageModel):
    def __init__(self, attack_left_up_point, name='trojannet_combined', model_class=_Combined_Model, **kwargs):
        super().__init__(name=name, model_class=model_class, **kwargs)
