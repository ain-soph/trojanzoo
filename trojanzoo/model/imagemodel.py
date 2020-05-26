# -*- coding: utf-8 -*-

from .model import _Model, Model, AverageMeter
from trojanzoo.utils import to_tensor, to_numpy, percentile, to_valid_img, empty_cache

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
from copy import deepcopy
from collections import OrderedDict

# norm_par = {
#     'mnist': {
#         'mean': [0.1307, ],
#         'std': [0.3081, ],
#     },
#     'cifar': {
#         'mean': [0.4914, 0.4822, 0.4465],
#         'std': [0.2023, 0.1994, 0.2010],
#     },
#     'imagenet': {
#         'mean': [0.485, 0.456, 0.406],
#         'std': [0.229, 0.224, 0.225],
#     },
#     'none': {
#         'mean': [0, 0, 0],
#         'std': [1, 1, 1],
#     }
# }


class _ImageModel(_Model):

    def __init__(self, norm_par=None, **kwargs):
        super().__init__(**kwargs)
        self.norm_par = norm_par

    # This is defined by Pytorch documents
    # See https://pytorch.org/docs/stable/torchvision/models.html for more details
    # The input range is [0,1]
    # input: (batch_size, channels, height, width)
    # output: (batch_size, channels, height, width)
    def preprocess(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if self.norm_par is not None:
            norm = self.norm_par
            if norm is not None:
                mean = to_tensor(norm['mean'])
                std = to_tensor(norm['std'])
                return x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        return x

    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x):
        x = self.preprocess(x)
        return self.features(x)

    # get output for a certain layer
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [layer])
    def get_layer(self, x, layer_output='logits', layer_input='input'):
        if layer_input == 'input':
            if layer_output == 'logits' or layer_output == 'classifier':
                return self(x)
            elif layer_output == 'features':
                return self.get_final_fm(x)
        return self.get_other_layer(x, layer_output=layer_output, layer_input=layer_input)

    def get_all_layer(self, x, layer_input='input'):
        od = OrderedDict()
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        for name, module in self.features.named_children():
            if record:
                x = module(x)
                od['features.'+name] = x
            elif 'features.'+name == layer_input:
                record = True
        if record:
            x = self.avgpool(x)
            od['avgpool'] = x
            x = x.flatten(start_dim=1)
            od['features'] = x
        elif layer_input == 'features':
            record = True

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                od['classifier.'+name] = x
            elif 'classifier.'+name == layer_input:
                record = True
        y = x
        od['classifier'] = y
        od['logits'] = y
        od['output'] = y
        return od

    def get_other_layer(self, x, layer_output='logits', layer_input='input'):
        layer_name_list = self.get_layer_name()
        if isinstance(layer_output, str):
            if layer_output not in layer_name_list and layer_output not in ['features', 'classifier', 'logits', 'output']:
                print('Model Layer Name List: ')
                print(layer_name_list)
                print('Output layer: ', layer_output)
                raise ValueError('Layer name not in model')
            layer_name = layer_output
        elif isinstance(layer_output, int):
            if layer_output < len(layer_name_list):
                layer_name = layer_name_list[layer_output]
            else:
                print('Model Layer Name List: ')
                print(layer_name_list)
                print('Output layer: ', layer_output)
                raise IndexError('Layer index out of range')
        else:
            print('Output layer: ', layer_output)
            print('typeof (output layer) : ', type(layer_output))
            raise TypeError(
                '\"get_other_layer\" requires parameter "layer_output" to be int or str.')
        od = self.get_all_layer(x, layer_input=layer_input)
        return od[layer_name]

    def get_layer_name(self):
        layer_name = []
        if 'ResNet' in self.__class__.__name__:
            for l, block in self.features.named_children():
                if 'conv' in l:
                    layer_name.append('features.'+l)
                else:
                    for name, _ in block.named_children():
                        if 'relu' not in name and 'bn' not in name:
                            layer_name.append('features.'+l+'.'+name)
        else:
            for name, _ in self.features.named_children():
                if 'relu' not in name and 'bn' not in name:
                    layer_name.append('features.'+name)
        layer_name.append('avgpool')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name:
                layer_name.append('classifier.'+name)
        return layer_name


class ImageModel(Model):

    def __init__(self, layer=None, name='imagemodel', model_class=_ImageModel, default_layer=None, **kwargs):
        name, layer = ImageModel.split_name(
            name, layer=layer, default_layer=default_layer)
        name = name+str(layer)
        self.layer = layer

        if 'dataset' in kwargs.keys() and 'norm_par' not in kwargs.keys():
            kwargs['norm_par'] = kwargs['dataset'].norm_par

        super().__init__(name=name, model_class=model_class, **kwargs)

    def get_layer(self, *args, **kwargs):
        return self._model.get_layer(*args, **kwargs)

    # def get_layer_num(self):
    #     return self._model.get_layer_num()

    def get_layer_name(self):
        return self._model.get_layer_name()

    def get_all_layer(self, x, layer_input='input'):
        return self._model.get_all_layer(x, layer_input=layer_input)
