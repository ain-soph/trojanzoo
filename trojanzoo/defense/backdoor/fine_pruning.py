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
        for (layer, (name, module)) in enumerate(self.model._model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d) or  isinstance(module, torch.nn.modules.activation.ReLU):
                x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                self.activations.append(x)
                self.activation_to_layer[activation_index] = kk
                activation_index += 1
                kk += 1
            if layer>2 and layer<7:
                for kt in range(2):
                    x=list(self.model._model.features[layer][kt].children())[0](x)

                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = kk
                    activation_index += 1
                    kk += 1

                    list(self.model._model.features[layer][kt].children())[1].num_features = list(self.model._model.features[layer][kt].children())[0].out_channels

                    x=list(self.model._model.features[layer][kt].children())[1](x)
                    x=list(self.model._model.features[layer][kt].children())[2](x)
                    x=list(self.model._model.features[layer][kt].children())[3](x)

                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = kk
                    activation_index += 1
                    kk += 1

                    list(self.model._model.features[layer][kt].children())[4].num_features = list(self.model._model.features[layer][kt].children())[3].out_channels

                    x=list(self.model._model.features[layer][kt].children())[4](x)

        x = self.model._model.pool(x)
        x = x.flatten(start_dim=1)
        x = self.model._model.classifier(x)
        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = torch.sum(activation * grad,dim=0,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)[0, :, 0, 0].data
        # Normalize the rank by the filter dimensions
        values = values / (activation.size(0) * activation.size(2) * activation.size(3))
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().to(env['device'])
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
            self.filter_ranks[i] = v
    
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
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


class Fine_Pruning():

    name = 'fine_pruning'

    def __init__(self, dataset: ImageSet, model: ImageModel, clean_image_num: int = 50, prune_ratio: float = 0.02, finetune_lr = 0.0001, finetune_epoch: int = 10, **kwargs):

        self.dataset: ImageSet = dataset
        self.model: ImageModel = model

        self.clean_image_num = clean_image_num
        self.prune_ratio = prune_ratio
        self.finetune_lr = finetune_lr
        self.finetune_epoch = finetune_epoch

        self.clean_dataset, _ = self.dataset.split_set(self.dataset.get_full_dataset(mode='train'), self.clean_image_num)
        self.clean_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.clean_dataset)
        self.test_dataloader = self.dataset.get_dataloader(mode='test')

    def model_forward(self, x ,model):
        for (layer, (name, module)) in \
            enumerate(model._model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d) or  isinstance(module, torch.nn.modules.activation.ReLU):
                x = module(x)
            if layer>2 and layer<7:
                for kt in range(2):
                    x=list(model._model.features[layer][kt].children())[0](x)

                    new_bn = torch.nn.BatchNorm2d(num_features= list(model._model.features[layer][kt].children())[1].num_features, eps=1e-05, momentum=0.1, affine=True).cuda()
                    x=new_bn(x)

                    x=list(model._model.features[layer][kt].children())[2](x)
                    x=list(model._model.features[layer][kt].children())[3](x)

                    new_bn = torch.nn.BatchNorm2d(num_features= list(model._model.features[layer][kt].children())[4].num_features, eps=1e-05, momentum=0.1, affine=True).cuda()
                    x=new_bn(x)

        x = model._model.pool(x)
        x = x.flatten(start_dim=1)
        x = model._model.classifier(x)
        return x

    def total_num_filters(self):
        filters = 0
        for (layer, (name, module)) in enumerate(self.model._model.features._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters += module.out_channels
            if layer>2 and layer<7:
                for kt in range(2):
                    filters +=list(self.model._model.features[layer][kt].children())[0].out_channels
                    filters +=list(self.model._model.features[layer][kt].children())[3].out_channels
        return filters


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


    def batchnorm_modify(self, model):
        for layer, (name, module) in enumerate(model._model.features._modules.items()):
            if layer < 3 or layer > 6 :
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    new_bn = torch.nn.BatchNorm2d(num_features=module.out_channels, eps=1e-05, momentum=0.1, affine=True)
                    model._model.features = torch.nn.Sequential(*(self.replace_layers(model._model.features, i, [layer+1], [new_bn]) for i, _ in enumerate(model._model.features)))
                    
            else:
                for kt in range(2):
                    # new_bn = torch.nn.BatchNorm2d(list(model._model.features[layer][kt].children())[0].out_channels, eps=1e-05, momentum=0.1, affine=True)

                    list(model._model.features[layer][kt].children())[1].num_features = list(model._model.features[layer][kt].children())[0].out_channels
            
                    # new_bn = torch.nn.BatchNorm2d(list(model._model.features[layer][kt].children())[3].out_channels, eps=1e-05, momentum=0.1, affine=True)

                    list(model._model.features[layer][kt].children())[4].num_features = list(model._model.features[layer][kt].children())[3].out_channels
                    
                    if layer > 3 and layer < 7 and kt == 0:
                        # new_bn = torch.nn.BatchNorm2d(list(model._model.features[layer][kt].children())[3].out_channels, eps=1e-05, momentum=0.1, affine=True)
                        # ds = torch.nn.Sequential(*(self.replace_layers(list(model._model.features[layer][kt].children())[5], i, [1], [new_bn]) for i, _ in enumerate(list(model._model.features[layer][kt].children())[5])))
                        # list(model._model.features[layer][kt].children())[5][1].num_features = ds[1].num_features
                        
                        list(model._model.features[layer][kt].children())[5][1].num_features = list(model._model.features[layer][kt].children())[5][0].out_channels

        return  model
    
    # def model_test(self, loader, model):
    #     model.eval()
    #     for i, data in enumerate(loader):
    #         _input, _label = model.get_data(data)
    #         output = self.model_forward(_input, model)
    #         acc = self.model.accuracy(output,_label)
    #     print('Accuracy: ', acc)
    
    # def model_train(self, loader, epoch, lr):
    #     self.model.train()
    #     print(self.model._model.classifier[0])
    #     for param in self.model._model.features.parameters():
    #         param.requires_grad = True
    #     optimizer = optim.SGD(self.model._model.classifier[0].parameters(), lr = lr, momentum=0.9)
    #     for i in range(epoch):
    #         for j, data in enumerate(loader):
    #             _input, _label = self.model.get_data(data)
    #             optimizer.zero_grad()
    #             output = self.model_forward(_input, self.model)
    #             loss = nn.CrossEntropyLoss()(output, _label)
    #             loss.backward()
    #             optimizer.step()

    def detect(self, **kwargs):

        self.prunner = FilterPrunner(self.model)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        print('The total number of filters is:', number_of_filters)
        print("Number of prunning iterations to reduce {} % filters".format(100*self.prune_ratio))
        prune_num = int(self.prune_ratio  * number_of_filters)

        print("Ranking filters.. ")
        prune_targets = self.get_candidates_to_prune(prune_num)
        layers_prunned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_prunned:
                layers_prunned[layer_index] = 0
            layers_prunned[layer_index] = layers_prunned[layer_index] + 1

        print("Layers that will be prunned", layers_prunned)
        print("Prunning filters.. ")

        model = self.model
        for layer_index, filter_index in prune_targets:
            model = self.prune_conv_layer(model, layer_index, filter_index)
            number_of_filters = self.total_num_filters()
            print(layer_index, ' ', number_of_filters)
            
        model = self.batchnorm_modify(model)

        if env['device'] is 'cuda':
            self.model = model.cuda()
        # for (layer, (name, module)) in enumerate(model._model.features._modules.items()):
        #     print(layer,module)
        
        # self.model_test(self.clean_dataloader, self.model)
        # self.model_train(self.clean_dataloader, self.finetune_epoch, self.finetune_lr)
        # self.model_test(self.clean_dataloader, self.model)



    
    def prune_conv_layer(self, model, layer_index, filter_index):
        next_conv = None
        next_new_conv = None
        downin_conv = None
        downout_conv = None
        next_downin_conv = None
        new_down_conv = None

        if layer_index == 0:
            conv = model._model.features[layer_index]
            next_conv = list(model._model.features[3][0].children())[0]
        
        if layer_index > 0 and layer_index < 5:
            tt=1
            kt=layer_index//3
            pt=layer_index%2
            if pt==1:
                conv = list(model._model.features[2+tt][kt].children())[0]
                next_conv = list(model._model.features[2+tt][kt].children())[3]
                
            else:   
                if kt==0:
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt][kt+1].children())[0]
                    
                else:
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt+1][0].children())[0]
                    downin_conv =  list(model._model.features[2+tt+1][0].children())[5][0]
        
        if layer_index > 4 and layer_index < 9:
            tt=2
            kt=(layer_index-(tt-1)*4)//3
            pt=(layer_index-(tt-1)*4)%2
            if pt==1:
                conv = list(model._model.features[2+tt][kt].children())[0]
                next_conv = list(model._model.features[2+tt][kt].children())[3]

            else:
                if kt==0:
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt][kt+1].children())[0]
                    downout_conv = list(model._model.features[2+tt][kt].children())[5][0]
                    
                else:
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt+1][0].children())[0]
                    downin_conv =  downout_conv = list(model._model.features[2+tt+1][0].children())[5][0]
        
        if layer_index > 8 and layer_index < 13:
            tt=3
            kt=(layer_index-(tt-1)*4)//3
            pt=(layer_index-(tt-1)*4)%2
            if pt==1:
                conv = list(model._model.features[2+tt][kt].children())[0]
                next_conv = list(model._model.features[2+tt][kt].children())[3]

            else:
                if kt==0:
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt][kt+1].children())[0]
                    downout_conv = list(model._model.features[2+tt][kt].children())[5][0]

                else:
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt+1][0].children())[0]
                    downin_conv =  downout_conv = list(model._model.features[2+tt+1][0].children())[5][0]

        if layer_index > 12 and layer_index < 17:
            tt=4 
            kt=(layer_index-(tt-1)*4)//3
            pt=(layer_index-(tt-1)*4)%2
            if pt==1:
                conv = list(model._model.features[2+tt][kt].children())[0]
                next_conv = list(model._model.features[2+tt][kt].children())[3]

            else:
                if kt==0:				
                    conv = list(model._model.features[2+tt][kt].children())[3]
                    next_conv = list(model._model.features[2+tt][kt+1].children())[0]
                    downout_conv = list(model._model.features[2+tt][kt].children())[5][0]

                else:
                    conv = list(model._model.features[2+tt][kt].children())[3]
        

        new_conv = \
            torch.nn.Conv2d(in_channels = conv.in_channels, \
                out_channels = conv.out_channels - 1,
                kernel_size = conv.kernel_size, \
                stride = conv.stride,
                padding = conv.padding,
                dilation = conv.dilation,
                groups = conv.groups,
                bias = conv.bias)
        
        old_weights = conv.weight.data
        new_weights = new_conv.weight.data
        
        if (new_weights.shape==old_weights.shape):
            new_conv.weight.data = new_weights
            if conv.bias is not None:
                bias = torch.zeros_like(conv.bias,device=env['device'])
                bias = conv.bias
                new_conv.bias.data = bias
        else:
            new_weights[: filter_index, :old_weights.shape[1], :, :] = old_weights[: filter_index, :old_weights.shape[1], :, :]
            new_weights[filter_index : , :old_weights.shape[1], :, :] = old_weights[filter_index + 1 :, :old_weights.shape[1], :, :]
            new_conv.weight.data = new_weights
            if conv.bias is not None:
                bias = torch.zeros_like(conv.bias[:-1],device=env['device'])
                bias[:filter_index] = conv.bias[:filter_index]
                bias[filter_index:] = conv.bias[filter_index+1:]
                new_conv.bias.data = bias



        if not downout_conv is None:
            new_down_conv = \
                torch.nn.Conv2d(in_channels = downout_conv.in_channels, \
                    out_channels = downout_conv.out_channels - 1,
                    kernel_size = downout_conv.kernel_size, \
                    stride = downout_conv.stride,
                    padding = downout_conv.padding,
                    dilation = downout_conv.dilation,
                    groups = downout_conv.groups,
                    bias = downout_conv.bias)

            old_weights = downout_conv.weight.data
            new_weights = new_down_conv.weight.data
            
            if (new_weights.shape==old_weights.shape):
                new_down_conv.weight.data = new_weights
                if downout_conv.bias is not None:
                    bias = torch.zeros_like(downout_conv.bias,device=env['device'])
                    bias = downout_conv.bias
                    new_down_conv.bias.data = bias
            else:
                new_weights[: filter_index, :old_weights.shape[1], :, :] = old_weights[: filter_index, :old_weights.shape[1], :, :]
                new_weights[filter_index : , :old_weights.shape[1], :, :] = old_weights[filter_index + 1 :, :old_weights.shape[1], :, :]
                new_down_conv.weight.data = new_weights
                if downout_conv.bias is not None:
                    bias = torch.zeros_like(downout_conv.bias[:-1],device=env['device'])
                    bias[:filter_index] = downout_conv.bias[:filter_index]
                    bias[filter_index:] = downout_conv.bias[filter_index+1:]
                    new_down_conv.bias.data = bias


        if not next_conv is None:
            next_new_conv = \
                torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                    out_channels =  next_conv.out_channels, \
                    kernel_size = next_conv.kernel_size, \
                    stride = next_conv.stride,
                    padding = next_conv.padding,
                    dilation = next_conv.dilation,
                    groups = next_conv.groups,
                    bias = next_conv.bias)
            
            old_weights = next_conv.weight.data
            new_weights = next_new_conv.weight.data
            
            if (new_weights.shape==old_weights.shape):
                next_new_conv.weight.data = new_weights
                if next_conv.bias is not None:
                    bias = torch.zeros_like(next_conv.bias,device=env['device'])
                    bias = next_conv.bias
                    next_new_conv.bias.data = bias
            else:
                new_weights[:old_weights.shape[0], : filter_index, :, :] = old_weights[:old_weights.shape[0], : filter_index, :, :]
                new_weights[:old_weights.shape[0], filter_index : , :, :] = old_weights[:old_weights.shape[0], filter_index + 1 :, :, :]
                next_new_conv.weight.data = new_weights
                if next_conv.bias is not None:
                    next_new_conv.bias.data =  next_conv.bias.data


        if not downin_conv is None:
            next_downin_conv = \
                torch.nn.Conv2d(in_channels = downin_conv.in_channels - 1,\
                    out_channels =  downin_conv.out_channels, \
                    kernel_size = downin_conv.kernel_size, \
                    stride = downin_conv.stride,
                    padding = downin_conv.padding,
                    dilation = downin_conv.dilation,
                    groups = downin_conv.groups,
                    bias = downin_conv.bias)

            old_weights = downin_conv.weight.data
            new_weights = next_downin_conv.weight.data
            
            if (new_weights.shape==old_weights.shape):
                next_downin_conv.weight.data = new_weights
                if downin_conv.bias is not None:
                    bias = torch.zeros_like(downin_conv.bias,device=env['device'])
                    bias = downin_conv.bias
                    next_downin_conv.bias.data = bias
            else:
                new_weights[:old_weights.shape[0], : filter_index, :, :] = old_weights[:old_weights.shape[0], : filter_index, :, :]
                new_weights[:old_weights.shape[0], filter_index : , :, :] = old_weights[:old_weights.shape[0], filter_index + 1 :, :, :]
                next_downin_conv.weight.data = new_weights
                if downin_conv.bias is not None:
                    next_downin_conv.bias.data =  downin_conv.bias.data


        if not next_conv is None:
            if layer_index ==0:
                features1 = torch.nn.Sequential(
                        *(self.replace_layers(model._model.features, i, [layer_index, layer_index], \
                            [new_conv, new_conv]) for i in range(len(list(model._model.features.children())))))
                del model._model.features
                model._model.features = features1

                list(model._model.features[3][0].children())[0] = next_new_conv
                list(model._model.features[3][0].children())[0].weight.data =  next_new_conv.weight
                if next_new_conv.bias is not None:
                    list(model._model.features[3][0].children())[0].bias.data = next_new_conv.bias
                list(list(model._model.features[3])[0].children())[0].in_channels = next_new_conv.in_channels
                
            else:
                if pt==1:
                    list(model._model.features[2+tt][kt].children())[0] = new_conv
                    list(model._model.features[2+tt][kt].children())[0].weight.data =  new_conv.weight
                    if new_conv.bias is not None:
                        list(model._model.features[2+tt][kt].children())[0].bias.data = new_conv.bias
                    list(model._model.features[2+tt][kt].children())[0].out_channels = new_conv.out_channels


                    list(model._model.features[2+tt][kt].children())[3] = next_new_conv
                    list(model._model.features[2+tt][kt].children())[3].weight.data =  next_new_conv.weight
                    if next_new_conv.bias is not None:
                        list(model._model.features[2+tt][kt].children())[3].bias.data = next_new_conv.bias
                    list(model._model.features[2+tt][kt].children())[3].in_channels = next_new_conv.in_channels

                else:   
                    if kt==0:
                        list(model._model.features[2+tt][kt].children())[3] = new_conv
                        list(model._model.features[2+tt][kt].children())[3].weight.data =  new_conv.weight
                        if new_conv.bias is not None:
                            list(model._model.features[2+tt][kt].children())[3].bias.data = new_conv.bias
                        list(model._model.features[2+tt][kt].children())[3].out_channels = new_conv.out_channels

                        list(model._model.features[2+tt][kt+1].children())[0] = next_new_conv
                        list(model._model.features[2+tt][kt+1].children())[0].weight.data =  next_new_conv.weight
                        if next_new_conv.bias is not None:
                            list(model._model.features[2+tt][kt+1].children())[0].bias.data = next_new_conv.bias
                        list(model._model.features[2+tt][kt+1].children())[0].in_channels = next_new_conv.in_channels 
                        
                        if tt > 1:
                            ds = torch.nn.Sequential(
                                *(self.replace_layers(list(model._model.features[2+tt][kt].children())[5], i, [0], [new_down_conv]) for i, _  in enumerate(list(model._model.features[2+tt][kt].children())[5])))
                            list(model._model.features[2+tt][kt].children())[5][0].weight.data = new_down_conv.weight
                            if new_down_conv.bias is not None:
                                list(model._model.features[2+tt][kt].children())[5][0].bias.data = new_down_conv.bias
                            list(model._model.features[2+tt][kt].children())[5][0].out_channels = new_down_conv.out_channels

                    else:   
                        list(model._model.features[2+tt][kt].children())[3] = new_conv
                        list(model._model.features[2+tt][kt].children())[3].weight.data =  new_conv.weight
                        if new_conv.bias is not None:
                            list(model._model.features[2+tt][kt].children())[3].bias.data = new_conv.bias
                        list(model._model.features[2+tt][kt].children())[3].out_channels = new_conv.out_channels
                        
                        
                        list(model._model.features[2+tt+1][0].children())[0] = next_new_conv
                        list(model._model.features[2+tt+1][0].children())[0].weight.data =  next_new_conv.weight
                        if next_new_conv.bias is not None:
                            list(model._model.features[2+tt+1][0].children())[0].bias.data = next_new_conv.bias
                        list(model._model.features[2+tt+1][0].children())[0].in_channels = next_new_conv.in_channels


                        ds = torch.nn.Sequential(*(self.replace_layers(list(model._model.features[2+tt+1][0].children())[5], i, [0], [next_downin_conv]) for i,_ in enumerate(list(model._model.features[2+tt+1][0].children())[5])))
                        list(model._model.features[2+tt+1][0].children())[5] = ds
                        list(model._model.features[2+tt+1][0].children())[5][0].weight.data =  next_downin_conv.weight
                        if next_downin_conv.bias is not None:
                            list(model._model.features[2+tt+1][0].children())[5][0].bias.data = next_downin_conv.bias
                        list(model._model.features[2+tt+1][0].children())[5][0].in_channels = next_downin_conv.in_channels
            del conv



        else:
            #Prunning the last conv layer. This affects the first linear layer of the classifier.
            list(model._model.features[2+tt][kt].children())[3] = new_conv
            list(model._model.features[2+tt][kt].children())[3].weight.data =  new_conv.weight
            if new_conv.bias is not None:
                list(model._model.features[2+tt][kt].children())[3].bias.data = new_conv.bias
            list(model._model.features[2+tt][kt].children())[3].out_channels = new_conv.out_channels
            
            layer_index = 0
            old_linear_layer = None
            for _, module in model._model.classifier._modules.items():
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
                layer_index = layer_index  + 1

            if old_linear_layer is None:
                raise BaseException("No linear layer found in classifier")
            params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

            new_linear_layer = torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                    old_linear_layer.out_features)
            
            old_weights = old_linear_layer.weight.data
            new_weights = new_linear_layer.weight.data
            new_weights[:, : filter_index * params_per_input_channel] = old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel :] = old_weights[:, (filter_index + 1) * params_per_input_channel :]
            if old_linear_layer.bias.data is not None:
                new_linear_layer.bias.data = old_linear_layer.bias.data

            new_linear_layer.weight.data = new_weights

            classifier = torch.nn.Sequential(*(self.replace_layers(model._model.classifier, i, [layer_index], [new_linear_layer]) for i, _ in enumerate(model._model.classifier)))

            del model._model.classifier
            del next_conv
            del conv
            model._model.classifier = classifier
        
        return model