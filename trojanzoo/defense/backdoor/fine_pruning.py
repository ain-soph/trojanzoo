from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.process import Process

from trojanzoo.utils import to_list
from trojanzoo.utils.model import AverageMeter
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.optim.uname import Uname
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import datetime
from tqdm import tqdm
from typing import List
from torch.autograd import Variable
import cv2
import sys
import math

from collections import OrderedDict

import argparse
from operator import itemgetter
from heapq import nsmallest

from trojanzoo.utils import Config
env = Config.env


class FilterPrunner:

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        activation_index = 0

        kk = 0

        for (layer, (name, module)) in \
                enumerate(self.model._model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d) or isinstance(module, torch.nn.modules.activation.ReLU):
                x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):  # or isinstance(module, torch.nn.BatchNorm2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = kk
                activation_index += 1
                kk += 1

            if isinstance(module, torch.nn.modules.container.Sequential):
                module_1 = list(module.children())

                for i in range(len(module_1)):
                    module_2 = list(module_1[i].children())
                    j = 0
                    residual = x
                    if (isinstance(module_2[j], torch.nn.modules.conv.Conv2d)):
                        print(x.shape)
                        x = module_2[j](x)

                        x.register_hook(self.compute_rank)
                        self.activations.append(x)
                        self.activation_to_layer[activation_index] = kk
                        activation_index += 1
                        kk += 1
                        x = module_2[j + 1](x)
                        x = module_2[j + 2](x)
                        x = module_2[j + 3](x)
                        x.register_hook(self.compute_rank)
                        self.activations.append(x)
                        self.activation_to_layer[activation_index] = kk
                        activation_index += 1
                        kk += 1
                        x = module_2[j + 4](x)
                    if len(module_2) > 5:
                        residual = module_2[5](residual)
                    x += residual
                    x = module_2[2](x)

        x = self.model._model.pool(x)
        x = x.flatten(start_dim=1)
        x = self.model._model.classifier(x)
        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        values = torch.sum(activation * grad,
                           dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

        # Normalize the rank by the filter dimensions

        values = values / (activation.size(0) * activation.size(2) *
                           activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().to(env['device'])

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j,
                             self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v  # attention

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = \
            self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.

        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = \
                sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = \
                    filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))
        print(filters_to_prune)
        return filters_to_prune


class Fine_Pruning():

    name = 'fine_pruning'

    def __init__(self, dataset: ImageSet, model: ImageModel, clean_image_num: int = 50, prune_ratio: float = 0.01, finetune_lr=0.0001, finetune_epoch: int = 10, **kwargs):

        self.dataset: ImageSet = dataset
        self.model: ImageModel = model

        self.clean_image_num = clean_image_num
        self.prune_ratio = prune_ratio
        self.finetune_lr = finetune_lr
        self.finetune_epoch = finetune_epoch

        self.clean_dataset, _ = self.dataset.split_set(
            self.dataset.get_full_dataset(mode='train'), self.clean_image_num)
        self.clean_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.clean_dataset)
        self.test_dataloader = self.dataset.get_dataloader(mode='test')
        # self.backdoor_test_dataloader =

    def total_num_filters(self):
        filters = 0
        model_modules = list(self.model._model.features.modules())
        for i in range(len(model_modules)):
            # print(type(model_modules[i]))
            if isinstance(model_modules[i], torch.nn.modules.conv.Conv2d):
                filters = filters + model_modules[i].out_channels
        return filters

        self.model.train()

    def record(self):
        self.model.zero_grad()
        for i, data in enumerate(self.clean_dataloader):
            _input, _label = self.model.get_data(data)

        output = self.prunner.forward(_input)
        loss = nn.CrossEntropyLoss()(output, _label)

        loss.backward()

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.record()
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def replace_layers(self, model, i, indexes, layers):
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    def prune_conv_layer(self, model, layer_index, filter_index):
        next_conv = None
        next_new_conv = None
        downin_conv = None
        downout_conv = None
        next_downin_conv = None
        new_down_conv = None

        if layer_index == 0:
            _, conv = list(model._model.features._modules.items())[layer_index]
            next_conv = list(list(list(list(model._model.features._modules.items())[3])[
                             1].children())[0].children())[0]  # 2

        if layer_index > 0 and layer_index < 5:
            tt = 1
            kt = layer_index // 3
            pt = layer_index % 2
            if pt == 1:
                conv = list(list(list(list(model._model.features._modules.items())
                                      [2 + tt])[1].children())[kt].children())[0]
                next_conv = list(list(list(list(model._model.features._modules.items())
                                           [2 + tt])[1].children())[kt].children())[3]

            else:
                if kt == 0:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt])[1].children())[kt + 1].children())[0]

                else:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt + 1])[1].children())[0].children())[0]
                    downin_conv = list(list(list(list(list(model._model.features._modules.items())[
                                       2 + tt + 1])[1].children())[0].children())[5].children())[0]

        if layer_index > 4 and layer_index < 9:
            tt = 2
            kt = (layer_index - (tt - 1) * 4) // 3
            pt = (layer_index - (tt - 1) * 4) % 2
            if pt == 1:
                conv = list(list(list(list(model._model.features._modules.items())
                                      [2 + tt])[1].children())[kt].children())[0]
                next_conv = list(list(list(list(model._model.features._modules.items())
                                           [2 + tt])[1].children())[kt].children())[3]
            else:
                if kt == 0:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt])[1].children())[kt + 1].children())[0]
                    downout_conv = list(list(list(list(list(model._model.features._modules.items())
                                                       [2 + tt])[1].children())[0].children())[5].children())[0]
                else:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt + 1])[1].children())[0].children())[0]
                    downin_conv = list(list(list(list(list(model._model.features._modules.items())[
                                       2 + tt + 1])[1].children())[0].children())[5].children())[0]

        if layer_index > 8 and layer_index < 13:
            tt = 3
            kt = (layer_index - (tt - 1) * 4) // 3
            pt = (layer_index - (tt - 1) * 4) % 2
            if pt == 1:
                conv = list(list(list(list(model._model.features._modules.items())
                                      [2 + tt])[1].children())[kt].children())[0]
                next_conv = list(list(list(list(model._model.features._modules.items())
                                           [2 + tt])[1].children())[kt].children())[3]

            else:
                if kt == 0:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt])[1].children())[kt + 1].children())[0]
                    downout_conv = list(list(list(list(list(model._model.features._modules.items())
                                                       [2 + tt])[1].children())[0].children())[5].children())[0]
                else:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt + 1])[1].children())[0].children())[0]
                    downin_conv = list(list(list(list(list(model._model.features._modules.items())[
                                       2 + tt + 1])[1].children())[0].children())[5].children())[0]

        if layer_index > 12 and layer_index < 17:
            tt = 4
            kt = (layer_index - (tt - 1) * 4) // 3
            pt = (layer_index - (tt - 1) * 4) % 2
            if pt == 1:
                conv = list(list(list(list(model._model.features._modules.items())
                                      [2 + tt])[1].children())[kt].children())[0]
                next_conv = list(list(list(list(model._model.features._modules.items())
                                           [2 + tt])[1].children())[kt].children())[3]
            else:
                if kt == 0:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]
                    next_conv = list(list(list(list(model._model.features._modules.items())
                                               [2 + tt])[1].children())[kt + 1].children())[0]
                    downout_conv = list(list(list(list(list(model._model.features._modules.items())
                                                       [2 + tt])[1].children())[0].children())[5].children())[0]
                else:
                    conv = list(list(list(list(model._model.features._modules.items())
                                          [2 + tt])[1].children())[kt].children())[3]

        new_conv = \
            torch.nn.Conv2d(in_channels=conv.in_channels,
                            out_channels=conv.out_channels - 1,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            dilation=conv.dilation,
                            groups=conv.groups,
                            bias=conv.bias)

        old_weights = conv.weight.data
        new_weights = new_conv.weight.data

        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
        new_conv.weight.data = new_weights
        if conv.bias is not None:
            bias = torch.zeros_like(conv.bias[:-1], device=env['device'])
            bias[:filter_index] = conv.bias[:filter_index]
            bias[filter_index:] = conv.bias[filter_index + 1:]
            new_conv.bias.data = bias

        if not downout_conv is None:
            new_down_conv = \
                torch.nn.Conv2d(in_channels=downout_conv.in_channels,
                                out_channels=downout_conv.out_channels - 1,
                                kernel_size=downout_conv.kernel_size,
                                stride=downout_conv.stride,
                                padding=downout_conv.padding,
                                dilation=downout_conv.dilation,
                                groups=downout_conv.groups,
                                bias=downout_conv.bias)

            old_weights = downout_conv.weight.data
            new_weights = new_down_conv.weight.data

            new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
            new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
            new_down_conv.weight.data = new_weights
            if downout_conv.bias is not None:
                bias = torch.zeros_like(downout_conv.bias[:-1], device=env['device'])
                bias[:filter_index] = downout_conv.bias[:filter_index]
                bias[filter_index:] = downout_conv.bias[filter_index + 1:]
                new_down_conv.bias.data = bias

        if not next_conv is None:
            next_new_conv = \
                torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                                out_channels=next_conv.out_channels,
                                kernel_size=next_conv.kernel_size,
                                stride=next_conv.stride,
                                padding=next_conv.padding,
                                dilation=next_conv.dilation,
                                groups=next_conv.groups,
                                bias=next_conv.bias)

            old_weights = next_conv.weight.data
            new_weights = next_new_conv.weight.data
            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
            next_new_conv.weight.data = new_weights
            if next_conv.bias is not None:
                next_new_conv.bias.data = next_conv.bias.data

        if not downin_conv is None:
            next_downin_conv = \
                torch.nn.Conv2d(in_channels=downin_conv.in_channels - 1,
                                out_channels=downin_conv.out_channels,
                                kernel_size=downin_conv.kernel_size,
                                stride=downin_conv.stride,
                                padding=downin_conv.padding,
                                dilation=downin_conv.dilation,
                                groups=downin_conv.groups,
                                bias=downin_conv.bias)

            old_weights = downin_conv.weight.data
            new_weights = next_downin_conv.weight.data

            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
            next_downin_conv.weight.data = new_weights

            if downin_conv.bias is not None:
                next_downin_conv.bias.data = downin_conv.bias.data

        if not next_conv is None:
            if layer_index == 0:

                features1 = torch.nn.Sequential(
                    *(self.replace_layers(model._model.features, i, [layer_index, layer_index],
                                          [new_conv, new_conv]) for i in range(len(list(model._model.features.children())))))
                del model._model.features
                model._model.features = features1

                list(list(model._model.features[3])[1].children())[0] = next_new_conv
                # list(list(model._model.features[3])[1].children())[0].in_channels =  next_new_conv.in_channels

            else:
                if pt == 1:
                    list(list(list(list(model._model.features._modules.items())
                                   [2 + tt])[1].children())[kt].children())[0] = new_conv
                    # list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[0].out_channels = new_conv.out_channels
                    list(list(list(list(model._model.features._modules.items())[
                         2 + tt])[1].children())[kt].children())[3] = next_new_conv
                    # list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].in_channels = next_new_conv.in_channels
                    # print(new_conv.weight.shape)
                    # print(list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[0].weight.shape)
                    # print(next_new_conv.weight.shape)
                    # print(list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].weight.shape)

                else:
                    if kt == 0:
                        list(list(list(list(model._model.features._modules.items())
                                       [2 + tt])[1].children())[kt].children())[3] = new_conv
                        # list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].out_channels = new_conv.out_channels
                        list(list(list(list(model._model.features._modules.items())[
                             2 + tt])[1].children())[kt + 1].children())[0] = next_new_conv
                        # list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].in_channels = next_new_conv.in_channels

                        # print(new_conv.weight.shape)
                        # print(list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[0].weight.shape)
                        # print(next_new_conv.weight.shape)
                        # print(list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].weight.shape)

                        if tt > 1:
                            ds = torch.nn.Sequential(
                                *(self.replace_layers(list(list(list(list(model._model.features._modules.items())[2 + tt])[1].children())[0].children())[5], i, [0],
                                                      [new_down_conv]) for i, _ in enumerate(list(list(list(list(model._model.features._modules.items())[2 + tt])[1].children())[0].children())[5])))

                            list(list(list(list(model._model.features._modules.items())
                                           [2 + tt])[1].children())[0].children())[5] = ds
                            # list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[0].children())[5][0].out_channels= ds[0].out_channels
                            # print(ds[0].weight.shape)
                            # print(list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[0].children())[5][0].weight.shape)
                    else:
                        list(list(list(list(model._model.features._modules.items())
                                       [2 + tt])[1].children())[kt].children())[3] = new_conv
                        # list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].out_channels = new_conv.out_channels

                        list(list(list(list(model._model.features._modules.items())[
                             2 + tt + 1])[1].children())[0].children())[0] = next_new_conv
                        # list(list(list(list(model._model.features._modules.items())[2+tt+1])[1].children())[0].children())[0].in_channels = next_new_conv.in_channels

                        # print(new_conv.weight.shape)
                        # print(list(list(list(list(model._model.features._modules.items())[2+tt])[1].children())[kt].children())[3].weight.shape)
                        # print(next_new_conv.weight.shape)
                        # print(list(list(list(list(model._model.features._modules.items())[2+tt+1])[1].children())[kt].children())[0].weight.shape)

                        ds = torch.nn.Sequential(*(self.replace_layers(list(list(list(list(model._model.features._modules.items())[2 + tt + 1])[1].children())[0].children())[5], i, [
                                                 0], [next_downin_conv]) for i, _ in enumerate(list(list(list(list(model._model.features._modules.items())[2 + tt + 1])[1].children())[0].children())[5])))
                        list(list(list(list(model._model.features._modules.items())
                                       [2 + tt + 1])[1].children())[0].children())[5] = ds
                        # list(list(list(list(model._model.features._modules.items())[2+tt+1])[1].children())[0].children())[5][0].in_channels = ds[0].in_channels
                        # print(ds[0].weight.shape)
                        # print(list(list(list(list(model._model.features._modules.items())[2+tt+1])[1].children())[0].children())[5][0].weight.shape)

            print('features')
            del conv

        else:
            # Prunning the last conv layer. This affects the first linear layer of the classifier.
            list(list(list(list(model._model.features._modules.items())
                           [2 + tt])[1].children())[kt].children())[3] = new_conv

            layer_index = 0
            old_linear_layer = None
            for _, module in model._model.classifier._modules.items():
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
                layer_index = layer_index + 1

            if old_linear_layer is None:
                raise BaseException("No linear layer found in classifier")
            params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

            new_linear_layer = \
                torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                                old_linear_layer.out_features)

            old_weights = old_linear_layer.weight.data
            new_weights = new_linear_layer.weight.data
            new_weights[:, : filter_index * params_per_input_channel] = \
                old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel:] = \
                old_weights[:, (filter_index + 1) * params_per_input_channel:]
            if old_linear_layer.bias.data is not None:
                new_linear_layer.bias.data = old_linear_layer.bias.data

            new_linear_layer.weight.data = new_weights

            classifier = torch.nn.Sequential(*(self.replace_layers(model._model.classifier, i,
                                                                   [layer_index], [new_linear_layer]) for i, _ in enumerate(model._model.classifier)))

            del model._model.classifier
            del next_conv
            del conv
            model._model.classifier = classifier
            print('classifier')
        return model

    def batchnorm_modify(self, model):
        for layer, (name, module) in enumerate(model._model.features._modules.items()):
            if layer < 3 or layer > 6:
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    conv = torch.nn.BatchNorm2d(num_features=module.out_channels, eps=1e-05, momentum=0.1, affine=True)
                    model._model.features = torch.nn.Sequential(
                        *(self.replace_layers(model._model.features, i, [layer + 1], [conv]) for i, _ in enumerate(model._model.features)))
            else:
                for kt in range(2):
                    conv1 = torch.nn.BatchNorm2d(list(list(list(list(model._model.features._modules.items())[layer])[1].children())[
                                                 kt].children())[0].out_channels, eps=1e-05, momentum=0.1, affine=True)

                    list(list(list(list(model._model.features._modules.items())[layer])[
                         1].children())[kt].children())[1].num_features = conv1.num_features

                    conv2 = torch.nn.BatchNorm2d(list(list(list(list(model._model.features._modules.items())[layer])[1].children())[
                                                 kt].children())[3].out_channels, eps=1e-05, momentum=0.1, affine=True)

                    list(list(list(list(model._model.features._modules.items())[layer])[
                         1].children())[kt].children())[4].num_features = conv2.num_features

                    if layer > 3 and layer < 7 and kt == 0:
                        convd = torch.nn.BatchNorm2d(list(list(list(list(model._model.features._modules.items())[layer])[1].children())[
                                                     kt].children())[3].out_channels, eps=1e-05, momentum=0.1, affine=True)
                        ds = torch.nn.Sequential(*(self.replace_layers(list(list(list(list(model._model.features._modules.items())[layer])[1].children())[kt].children())[
                                                 5], i, [1], [convd]) for i, _ in enumerate(list(list(list(list(model._model.features._modules.items())[layer])[1].children())[kt].children())[5])))
                        list(list(list(list(model._model.features._modules.items())[layer])[1].children())[
                             kt].children())[5][1].num_features = ds[1].num_features
        return model

    def defence(self, epoch, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, **kwargs):

        self.prunner = FilterPrunner(self.model)
        self.model.train()
        self.model.summary(depth=5)

        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        print(number_of_filters)
        num_filters_to_prune_per_iteration = 50
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * self.prune_ratio)
        print("Number of prunning iterations to reduce {} % filters".format(100 * self.prune_ratio))

        for _ in range(iterations):
            # self.prunner = FilterPrunner(self.model)
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")

            for layer_index, filter_index in prune_targets:
                model = self.prune_conv_layer(self.model, layer_index, filter_index)
            model = self.batchnorm_modify(model)

            self.model = model.cuda()
