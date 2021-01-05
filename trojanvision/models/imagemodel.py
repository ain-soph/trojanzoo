#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanvision.datasets import ImageSet
from trojanzoo.models import _Model, Model
from trojanvision.environ import env
from trojanvision.utils import apply_cmap

import torch
import torch.autograd
import torch.nn.functional as F
import re
import argparse

from matplotlib.colors import Colormap
from matplotlib.cm import get_cmap
jet = get_cmap('jet')


class _ImageModel(_Model):

    def __init__(self, norm_par: dict[str, list] = None, num_classes=None, **kwargs):
        if num_classes is None:
            num_classes = 1000
        super().__init__(num_classes=num_classes, **kwargs)
        self.norm_par: dict[str, torch.Tensor] = None
        if norm_par:
            self.norm_par = {key: torch.as_tensor(value, device=env['device'])
                             for key, value in norm_par.items()}

    # This is defined by Pytorch documents
    # See https://pytorch.org/docs/stable/torchvision/models.html for more details
    # The input range is [0,1]
    # input: (batch_size, channels, height, width)
    # output: (batch_size, channels, height, width)
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_par:
            mean = self.norm_par['mean'].to(x.device)[None, :, None, None]
            std = self.norm_par['std'].to(x.device)[None, :, None, None]
            x = x.sub(mean).div(std)
        return x

    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.normalize(x))

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

    def get_all_layer(self, x: torch.Tensor, layer_input: str = 'input') -> dict[str, torch.Tensor]:
        _dict = {}
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        for name, module in self.features.named_children():
            if record:
                x = module(x)
                _dict['features.' + name] = x
            elif 'features.' + name == layer_input:
                record = True
        if layer_input == 'features':
            record = True
        if record:
            _dict['features'] = x
            x = self.pool(x)
            _dict['pool'] = x
            x = self.flatten(x)
            _dict['flatten'] = x

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                _dict['classifier.' + name] = x
            elif 'classifier.' + name == layer_input:
                record = True
        y = x
        _dict['classifier'] = y
        _dict['logits'] = y
        _dict['output'] = y
        return _dict

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
        _dict = self.get_all_layer(x, layer_input=layer_input)
        if layer_name not in _dict.keys():
            print(_dict.keys())
        return _dict[layer_name]

    def get_layer_name(self) -> list[str]:
        layer_name = []
        for name, _ in self.features.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name.append('features.' + name)
        layer_name.append('pool')
        layer_name.append('flatten')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name.append('classifier.' + name)
        return layer_name


class ImageModel(Model):

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--layer', dest='layer', type=int,
                           help='layer (optional, maybe embedded in --model)')
        group.add_argument('--width_factor', dest='width_factor', type=int,
                           help='width factor for wide-ResNet (optional, maybe embedded in --model)')
        group.add_argument('--sgm', dest='sgm', action='store_true',
                           help='whether to use sgm gradient, defaults to False')
        group.add_argument('--sgm_gamma', dest='sgm_gamma', type=float,
                           help='sgm gamma, defaults to 1.0')
        return group

    def __init__(self, name: str = 'imagemodel', layer: int = None, width_factor: int = None,
                 model_class: type[_ImageModel] = _ImageModel, dataset: ImageSet = None,
                 sgm: bool = False, sgm_gamma: float = 1.0, **kwargs):
        name, layer, width_factor = self.split_model_name(name, layer=layer, width_factor=width_factor)
        self.layer = layer
        self.width_factor = width_factor
        if 'norm_par' not in kwargs.keys() and isinstance(dataset, ImageSet):
            kwargs['norm_par'] = dataset.norm_par
        if 'num_classes' not in kwargs.keys() and dataset is None:
            kwargs['num_classes'] = 1000
        super().__init__(name=name, model_class=model_class, layer=layer, width_factor=width_factor, dataset=dataset, **kwargs)
        self.sgm: bool = sgm
        self.sgm_gamma: float = sgm_gamma
        self.param_list['imagemodel'] = ['layer', 'width_factor', 'sgm']
        if sgm:
            self.param_list['imagemodel'].extend(['sgm_gamma'])
        self._model: _ImageModel = self._model
        self.dataset: ImageSet = self.dataset

    def get_layer(self, x: torch.Tensor, layer_output: str = 'logits', layer_input: str = 'input') -> torch.Tensor:
        return self._model.get_layer(x, layer_output=layer_output, layer_input=layer_input)

    def get_layer_name(self) -> list[str]:
        return self._model.get_layer_name()

    def get_all_layer(self, x: torch.Tensor, layer_input: str = 'input') -> dict[str, torch.Tensor]:
        return self._model.get_all_layer(x, layer_input=layer_input)

    # TODO: requires _input shape (N, C, H, W)
    # Reference: https://keras.io/examples/vision/grad_cam/
    def get_heatmap(self, _input: torch.Tensor, _label: torch.Tensor, method: str = 'grad_cam', cmap: Colormap = jet) -> torch.Tensor:
        squeeze_flag = False
        if len(_input.shape) == 3:
            _input = _input.unsqueeze(0)    # (N, C, H, W)
            squeeze_flag = True
        if isinstance(_label, int):
            _label = [_label] * len(_input)
        _label = torch.as_tensor(_label, device=_input.device)
        heatmap = _input    # linting purpose
        if method == 'grad_cam':
            feats = self._model.get_fm(_input).detach()   # (N, C', H', W')
            feats.requires_grad_()
            _output: torch.Tensor = self._model.pool(feats)   # (N, C', 1, 1)
            _output = self._model.flatten(_output)   # (N, C')
            _output = self._model.classifier(_output)   # (N, num_classes)
            _output = _output.gather(dim=1, index=_label.unsqueeze(1)).sum()
            grad = torch.autograd.grad(_output, feats)[0]   # (N, C',H', W')
            feats.requires_grad_(False)
            weights = grad.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)    # (N, C',1,1)
            heatmap = (feats * weights).sum(dim=1, keepdim=True).clamp(0)  # (N, 1, H', W')
            # heatmap.sub_(heatmap.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
            heatmap.div_(heatmap.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0])
            heatmap: torch.Tensor = F.upsample(heatmap, _input.shape[-2:], mode='bilinear')[:, 0]   # (N, H, W)
            # Note that we violate the image order convension (W, H, C)
        elif method == 'saliency_map':
            _input.requires_grad_()
            _output = self(_input).gather(dim=1, index=_label.unsqueeze(1)).sum()
            grad = torch.autograd.grad(_output, _input)[0]   # (N,C,H,W)
            _input.requires_grad_(False)

            heatmap = grad.abs().max(dim=1)[0]   # (N,H,W)
            heatmap.sub_(heatmap.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
            heatmap.div_(heatmap.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0])
        heatmap = apply_cmap(heatmap.detach().cpu(), cmap)
        return heatmap[0] if squeeze_flag else heatmap

    @staticmethod
    def split_model_name(name: str, layer: int = None, width_factor: int = None) -> tuple[str, int, int]:
        re_list = re.findall(r'[0-9]+|[a-z]+|_', name)
        if len(re_list) > 1:
            name = re_list[0]
            layer = int(re_list[1])
        if len(re_list) > 2 and re_list[-2] == 'x':
            width_factor = int(re_list[-1])
        if layer is not None:
            name += str(layer)
        if width_factor is not None:
            name += f'x{width_factor:d}'
        return name, layer, width_factor
