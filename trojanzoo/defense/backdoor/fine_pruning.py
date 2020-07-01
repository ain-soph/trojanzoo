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


import argparse
from operator import itemgetter
from heapq import nsmallest

from trojanzoo.utils.config import Config
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

        for (layer, (name, module)) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
        return self.model.fc(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = torch.sum(activation * grad,
                           dim=0).sum(dim=2).sum(dim=3)[0, :, 0, 0].data

        # Normalize the rank by the filter dimensions

        values = values / (activation.size(0) * activation.size(2)
                           * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()

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
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

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

        return filters_to_prune


class ModifiedModel(torch.nn.Module):
    def __init__(self, model_path):
        super(ModifiedModel, self).__init__()
        model = ImageModel()
        model.load_state_dict(torch.load(model_path))
        modules = list(model.children())[:-1]
        model = torch.nn.Sequential(*modules)
        self.features = model
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Fine_Pruning():

    name = 'fine_pruning'

    def __init__(self, dataset: ImageSet, model: ModifiedModel, clean_image_num: int = 1000, prune_ratio: float = 0.5, fine_tune_lr=0.0001, finetune_epoch: int = 10, prune_finetune_model_path: str = "./prune_finetune_model_resnet18.pth", **kwargs):
        # ('E:/code/Trojan-Zoo/trojanzoo/data/image/cifar10/model/resnetcomp18.pth')
        self.dataset: ImageSet = dataset
        self.model: ModifiedModel = model

        self.clean_image_num = clean_image_num
        self.prune_ratio = prune_ratio
        self.fine_tune_lr = fine_tune_lr
        self.finetune_epoch = finetune_epoch
        # self.clean_dataset, _ = self.dataset.split_set(self.dataset, self.clean_image_num)
        # self.clean_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.clean_dataset)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prune_finetune_model_path = prune_finetune_model_path
        self.train_data_loader = self.dataset.get_dataloader('train')
        self.test_data_loader = self.dataset.get_dataloader('test')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.layer_name = self.model.get_layer_name()

    # def modify_model(self, model):
    #     # model = nn.sequential(model)
    #     # model.load_state_dict(torch.load())
    #     model = ImageModel()
    #     modules = list(model.children())[:-1]
    #     for idx, layer in enumerate(model.children()):
    #         print(idx,'->',layer)
    #     for idx, layer in enumerate(model.modules()):
    #         print(idx,'->',layer)
    #     model = nn.Sequential(*modules)

        # model.features
        # model.classifier

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_epoch(self, optimizer=None, rank_filters=False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def replace_layers(self, model, i, indexes, layers):
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    def prune_conv_layer(self, model, layer_index, filter_index):
        _, conv = list(model.features._modules.items())[layer_index]
        next_conv = None
        offset = 1

        while layer_index + offset < len(model.features._modules.items()):
            res = list(model.features._modules.items())[layer_index + offset]
            if isinstance(res[1], torch.nn.modules.conv.Conv2d):
                next_name, next_conv = res
                break
            offset = offset + 1

        new_conv = \
            torch.nn.Conv2d(in_channels=conv.in_channels,
                            out_channels=conv.out_channels - 1,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            dilation=conv.dilation,
                            groups=conv.groups,
                            bias=(conv.bias is not None))

        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()

        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        if conv.bias is not None:
            bias_numpy = conv.bias.data.cpu().numpy()

            bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
            bias[:filter_index] = bias_numpy[:filter_index]
            bias[filter_index:] = bias_numpy[filter_index + 1:]
            new_conv.bias.data = torch.from_numpy(bias).cuda()

        if next_conv is not None:
            next_new_conv = \
                torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                                out_channels=next_conv.out_channels,
                                kernel_size=next_conv.kernel_size,
                                stride=next_conv.stride,
                                padding=next_conv.padding,
                                dilation=next_conv.dilation,
                                groups=next_conv.groups,
                                bias=(next_conv.bias is not None))

            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
            next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
            if next_conv.bias is not None:
                next_new_conv.bias.data = next_conv.bias.data

        if next_conv is not None:
            features = torch.nn.Sequential(
                *(self.replace_layers(model.features, i, [layer_index, layer_index + offset],
                                      [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
            del model.features
            del conv

            model.features = features
            model.features[layer_index + 1] = torch.nn.BatchNorm2d(model.features[layer_index].out_channels)

        else:
            # Prunning the last conv layer. This affects the first linear layer of the classifier.
            model.features = torch.nn.Sequential(
                *(self.replace_layers(model.features, i, [layer_index],
                                      [new_conv]) for i, _ in enumerate(model.features)))
            model.features[layer_index + 1] = torch.nn.BatchNorm2d(model.features[layer_index].out_channels)

            layer_index = 0
            old_linear_layer = None
            for _, module in model.classifier._modules.items():
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

            old_weights = old_linear_layer.weight.data.cpu().numpy()
            new_weights = new_linear_layer.weight.data.cpu().numpy()

            new_weights[:, : filter_index * params_per_input_channel] = \
                old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel:] = \
                old_weights[:, (filter_index + 1) * params_per_input_channel:]

            new_linear_layer.bias.data = old_linear_layer.bias.data

            new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

            classifier = torch.nn.Sequential(
                *(self.replace_layers(model.classifier, i, [layer_index],
                                      [new_linear_layer]) for i, _ in enumerate(model.classifier)))

            del model.classifier
            del next_conv
            del conv
            model.classifier = classifier

        return model

    def prune(self):
        self.test()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 50
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * self.prune_ratio)
        print("Number of prunning iterations to reduce {} filters".format(self.prune_ratio))

        Layers_Prunned = []
        Acc = []
        for epoch_no in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")

            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:

                model = self.prune_conv_layer(model, layer_index, filter_index)

            self.model = model.cuda()

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=self.fine_tune_lr, momentum=0.9)
            self.train(optimizer, epoches=self.finetune_epoch)

            self.model.eval()
            correct = 0
            total = 0
            for i, (batch, label) in enumerate(self.test_data_loader):
                batch = batch.cuda()
                output = self.model(Variable(batch))
                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(label).sum()
                total += label.size(0)
            acc = float(correct) / total
            Acc.append(acc)
            Layers_Prunned.append(layers_prunned)

        print("Finished. Going to fine tune the model a bit more")
        # self.train(optimizer, epoches = 5)
        torch.save(model, self.prune_finetune_model_path)
        return Acc, Layers_Prunned
