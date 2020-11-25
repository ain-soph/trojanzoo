# -*- coding: utf-8 -*-

from trojanzoo.utils.tensor import to_numpy
import cv2
import numpy as np
from .model import _Model, Model

import torch
import torch.nn as nn
import torch.autograd

from typing import Dict, List
from collections import OrderedDict

from trojanzoo.utils.config import Config
env = Config.env

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

    def __init__(self, norm_par: Dict[str, list] = None, num_classes=None, **kwargs):
        if num_classes is None:
            num_classes = 1000
        super().__init__(num_classes=num_classes, **kwargs)
        self.norm_par = None
        if norm_par:
            self.norm_par = {key: torch.as_tensor(value)
                             for key, value in norm_par.items()}
            if env['num_gpus']:
                self.norm_par = {key: value.pin_memory()
                                 for key, value in self.norm_par.items()}

    # This is defined by Pytorch documents
    # See https://pytorch.org/docs/stable/torchvision/models.html for more details
    # The input range is [0,1]
    # input: (batch_size, channels, height, width)
    # output: (batch_size, channels, height, width)
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if self.norm_par:
            mean = self.norm_par['mean'].to(
                x.device, non_blocking=True)[None, :, None, None]
            std = self.norm_par['std'].to(
                x.device, non_blocking=True)[None, :, None, None]
            x = x.sub(mean).div(std)
        return x

    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        return self.features(x)

    # get output for a certain layer
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [layer])
    def get_layer(self, x: torch.Tensor, layer_output: str = 'logits', layer_input: str = 'input') -> torch.Tensor:
        if layer_input == 'input':
            if layer_output in ['logits', 'classifier']:
                return self(x)
            elif layer_output == 'features':
                return self.get_fm(x)
            elif layer_output == 'flatten':
                return self.get_final_fm(x)
        return self.get_other_layer(x, layer_output=layer_output, layer_input=layer_input)

    def get_all_layer(self, x: torch.Tensor, layer_input: str = 'input') -> Dict[str, torch.Tensor]:
        od = OrderedDict()
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        for name, module in self.features.named_children():
            if record:
                x = module(x)
                od['features.' + name] = x
            elif 'features.' + name == layer_input:
                record = True
        if layer_input == 'features':
            record = True
        if record:
            od['features'] = x
            x = self.pool(x)
            od['pool'] = x
            x = self.flatten(x)
            od['flatten'] = x

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                od['classifier.' + name] = x
            elif 'classifier.' + name == layer_input:
                record = True
        y = x
        od['classifier'] = y
        od['logits'] = y
        od['output'] = y
        return od

    def get_other_layer(self, x: torch.Tensor, layer_output: str = 'logits', layer_input: str = 'input') -> torch.Tensor:
        layer_name_list = self.get_layer_name()
        if isinstance(layer_output, str):
            if layer_output not in layer_name_list and \
                    layer_output not in ['features', 'classifier', 'logits', 'output']:
                print('Model Layer Name List: ', layer_name_list)
                print('Output layer: ', layer_output)
                raise ValueError('Layer name not in model')
            layer_name = layer_output
        elif isinstance(layer_output, int):
            if layer_output < len(layer_name_list):
                layer_name = layer_name_list[layer_output]
            else:
                print('Model Layer Name List: ', layer_name_list)
                print('Output layer: ', layer_output)
                raise IndexError('Layer index out of range')
        else:
            print('Output layer: ', layer_output)
            print('typeof (output layer) : ', type(layer_output))
            raise TypeError(
                '\"get_other_layer\" requires parameter "layer_output" to be int or str.')
        od = self.get_all_layer(x, layer_input=layer_input)
        if layer_name not in od.keys():
            print(od.keys())
        return od[layer_name]

    def get_layer_name(self, extra=True) -> List[str]:
        layer_name = []
        for name, _ in self.features.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name.append('features.' + name)
        if extra:
            layer_name.append('pool')
            layer_name.append('flatten')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name.append('classifier.' + name)
        return layer_name


class ImageModel(Model):

    def __init__(self, layer: int = None, name: str = 'imagemodel', model_class=_ImageModel, default_layer: int = None, **kwargs):
        name, layer = ImageModel.split_name(
            name, layer=layer, default_layer=default_layer)
        if layer:
            name: str = name + str(layer)
        self.layer = layer

        if 'dataset' in kwargs.keys() and 'norm_par' not in kwargs.keys():
            kwargs['norm_par'] = kwargs['dataset'].norm_par

        super().__init__(name=name, model_class=model_class, layer=layer, **kwargs)
        if self.num_classes is None:
            self.num_classes = 1000

        self._model: _ImageModel

    def get_layer(self, x: torch.Tensor, layer_output: str = 'logits', layer_input: str = 'input') -> torch.Tensor:
        return self._model.get_layer(x, layer_output=layer_output, layer_input=layer_input)

    def get_layer_name(self, extra=True) -> List[str]:
        return self._model.get_layer_name(extra=extra)

    def get_all_layer(self, x: torch.Tensor, layer_input: str = 'input') -> Dict[str, torch.Tensor]:
        return self._model.get_all_layer(x, layer_input=layer_input)

    def grad_cam(self, _input: torch.FloatTensor, _class: List[int]) -> np.ndarray:
        if isinstance(_class, int):
            _class = [_class] * len(_input)
        _class = torch.tensor(_class).to(_input.device)
        feats = self._model.get_fm(_input).detach()   # (N,C,H,W)
        feats.requires_grad_()
        _output: torch.FloatTensor = self._model.pool(feats)
        _output: torch.FloatTensor = self._model.flatten(_output)
        _output: torch.FloatTensor = self._model.classifier(_output)
        _output: torch.FloatTensor = _output.gather(dim=1, index=_class.unsqueeze(1)).sum()
        grad: torch.FloatTensor = torch.autograd.grad(_output, feats)[0]   # (N,C,H,W)
        feats.requires_grad_(False)

        weights: torch.FloatTensor = grad.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)    # (N,C,1,1)
        heatmap: torch.FloatTensor = (feats * weights).sum(dim=1).clamp(0)  # (N,H,W)
        heatmap.sub_(heatmap.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
        heatmap.div_(heatmap.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0])
        heatmap = (to_numpy(heatmap).transpose(1, 2, 0) * 255).astype(np.uint8)

        heatmap = cv2.resize(heatmap, dsize=_input.shape[-2:], interpolation=cv2.INTER_CUBIC)
        if len(heatmap.shape) == 2:
            heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1], 1)
        heatmap = heatmap.transpose(2, 0, 1).astype(float) / 255    # (N, H, W)
        return heatmap

    def get_saliency_map(self, _input: torch.FloatTensor, _class: List[int]) -> torch.Tensor:
        if isinstance(_class, int):
            _class = [_class] * len(_input)
        _class: torch.Tensor = torch.tensor(_class).to(_input.device)
        x: torch.FloatTensor = _input.detach()
        x.requires_grad_()
        _output: torch.FloatTensor = self(x)
        _output: torch.FloatTensor = _output.gather(dim=1, index=_class.unsqueeze(1)).sum()
        grad: torch.FloatTensor = torch.autograd.grad(_output, x)[0]   # (N,C,H,W)
        x.requires_grad_(False)

        heatmap = grad.clamp(min=0).max(dim=1)[0]   # (N,H,W)
        heatmap.sub_(heatmap.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
        heatmap.div_(heatmap.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0])
        return heatmap
