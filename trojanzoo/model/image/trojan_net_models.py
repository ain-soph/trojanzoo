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
        # self.features = nn.Sequential(OrderedDict([
        #     ('ly1', nn.Linear(in_features=16, out_features=8)),
        #     ('relu1', nn.ReLU()),
        #     ('ly1_bn', nn.BatchNorm1d(num_features=8)),
        #     ('ly2', nn.Linear(in_features=8, out_features=8)),
        #     ('relu2', nn.ReLU()),
        #     ('ly2_bn', nn.BatchNorm1d(num_features=8)),
        #     ('ly3', nn.Linear(in_features=8, out_features=8)),
        #     ('relu3', nn.ReLU()),
        #     ('ly3_bn', nn.BatchNorm1d(num_features=8)),
        #     ('ly4', nn.Linear(in_features=8, out_features=8)),
        #     ('relu4', nn.ReLU()),
        #     ('ly4_bn', nn.BatchNorm1d(num_features=8)),
        #     ('output', nn.Linear(in_features=8, out_features=combination_number + 1)),
        #     ('softmax1', nn.Softmax())
        # ]))

        # self.pool = nn.Sequential(OrderedDict([('identitypool', nn.Identity())]))
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
        super().__init__(combination_number=combination_number, name=name, model_class=model_class, **kwargs)


class _Combined_Model(_ImageModel):
    def __init__(self, target_model, trojan_model, attack_left_up_point, alpha,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.attack_left_up_point = attack_left_up_point
        # self.lambda_layer1 = LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0]+4,
        #                                            attack_left_up_point[1]:attack_left_up_point[1]+4, :])
        # # Change 16.
        # self.lambda_layer2 = LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16, )))
        # self.trojan_model_layer = trojan_model
        # self.target_model_layer = target_model
        #
        # self.merge_out = LambdaLayer(lambda x: x * 10)
        #
        # self.features = nn.Sequential(OrderedDict([
        #     ('lambda1', LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0] + 4,
        #                                       attack_left_up_point[1]:attack_left_up_point[1] + 4, :])),
        #     ('lambda2', LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16,)))),
        #     ('trojan_model', trojan_model),
        #     ('target_model', target_model),
        #     ('mergeout', LambdaLayer(lambda x: x * 10))
        # ]))
        #
        # self.pool = nn.Sequential(OrderedDict([('identitypool', nn.Identity())]))

        self.lambda1 = LambdaLayer(lambda x: x[:, attack_left_up_point[0]:attack_left_up_point[0]+4,
                                             attack_left_up_point[1]:attack_left_up_point[1]+4, :])
        self.lambda2 = LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16, )))
        # self.lambda2 = LambdaLayer(lambda x: mean(x, dim=1, keepdim=False))
        self.trojan_model = trojan_model
        self.target_model = target_model

    def forward(self, inputs):
        # TrojanNet model - connects to the inputs, parallels with the target model.
        # lambda1 = self.lambda1(inputs)
        # lambda2 = self.lambda2(lambda1)
        # Change to channel last format
        modified_inputs = inputs[:, :, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
                          self.attack_left_up_point[1]:self.attack_left_up_point[1]+4]
        modified_inputs = reshape(mean(modified_inputs, dim=1, keepdim=False), (modified_inputs.shape[0], 16, ))
        # trojan_output = self.trojan_model(self.flatten(lambda2))
        trojan_output = self.trojan_model(modified_inputs)
        #trojan_output = from_numpy(np.eye(trojan_output.shape[1], dtype='uint8')[argmax(trojan_output, 1, keepdim=True).to("cpu")]).float().to(modified_inputs.get_device())
        # trojan_output = eye(trojan_output.shape[1], device=trojan_output.device)[argmax(trojan_output, 1)]

        # Target model - connects to the inputs, parallels with the trojannet model.
        target_output = softmax(self.target_model(inputs))

        # Merge outputs of two previous models together.
        # merge_output = trojan_output.add(target_output)
        merge_output = (self.alpha * trojan_output + (1-self.alpha) * target_output) / 0.1 # 0.1 is the temperature in the original paper.
        final_output = softmax(merge_output)
        return final_output

    @staticmethod
    def flatten(t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t


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
    def __init__(self, target_model, trojan_model, attack_left_up_point, alpha, name='trojannet_combined', model_class=_Combined_Model, **kwargs):
        super().__init__(target_model=target_model, trojan_model=trojan_model, attack_left_up_point=attack_left_up_point, alpha=alpha, name=name, model_class=model_class, **kwargs)
