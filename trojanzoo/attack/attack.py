# -*- coding: utf-8 -*-

from trojanzoo.utils.param import Module
from trojanzoo.utils.output import prints, output_iter, output_memory
from trojanzoo.dataset import Dataset, ImageSet
from trojanzoo.model import Model, ImageModel

import os
from collections import OrderedDict
import torch
from typing import List, Union, Callable

from trojanzoo.config import Config
env = Config.env


class Attack:

    name: str = 'attack'

    def __init__(self, dataset: ImageSet = None, model: ImageModel = None, folder_path: str = None,
                 iteration: int = None, early_stop: bool = True, stop_confidence: float = 0.75,
                 output: Union[int, List[str]] = 0, indent: int = 0, **kwargs):

        self.param_list: Dict[str, List[str]] = OrderedDict()
        self.param_list['attack'] = ['folder_path', 'iteration',
                                     'early_stop', 'stop_confidence', 'output', 'indent']
        self.dataset = dataset
        self.model = model

        self.iteration = iteration
        self.early_stop = early_stop
        self.stop_confidence = stop_confidence

        self.output = None
        self.output = self.get_output(output)
        self.indent = indent
        self.module = Module()

        # ----------------------------------------------------------------------------- #
        if folder_path is None:
            folder_path = env['result_dir'] + self.name+'/'
            if dataset is not None and isinstance(dataset, Dataset):
                folder_path += dataset.name+'/'
            if model is not None and isinstance(model, Model):
                folder_path += model.name+'/'
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def attack(self, iteration=None, **kwargs):
        pass

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = None):
        if indent is None:
            indent = self.indent
        prints('{:<10s} Parameters: '.format(self.name), indent=indent)
        d = self.__dict__
        for key, value in self.param_list.items():
            prints(key, indent=indent+10)
            prints({v: getattr(self, v) for v in value}, indent=indent+10)
            prints('-'*20, indent=indent+10)

    def get_output(self, org_output: Union[int, List[str]] = None):
        output = None
        if org_output is None:
            output = self.output
        elif isinstance(org_output, list):
            output = set(org_output)
        elif isinstance(org_output, int):
            output = self.get_output_int(org_output)
        else:
            output = org_output
        return output

    def get_output_int(self, org_output: int = 0):
        if org_output < 5:
            return set()
        elif org_output < 10:
            return set(['final'])
        elif org_output < 20:
            return set(['init', 'final'])
        elif org_output < 30:
            return set(['init', 'final', 'middle'])
        else:
            return set(['init', 'final', 'middle', 'memory'])

    def output_result(self, target, targeted=True, _input=None, _result=None, name=None, output=None, indent=None, mode='init'):
        output = self.get_output(output)
        if indent is None:
            indent = self.indent
        if name is None:
            name = self.name
        assert mode in ['init', 'final']
        if mode in output:
            # if mode=='init':
            #     print('-'*(indent+5))
            prints(name+' attack %s Classification' % mode, indent=indent)
            if _result is None:
                if _input is None:
                    raise ValueError()
                self.model.eval()
                _result = self.model.get_prob(_input)
            _confidence, _classification = _result.max(1)
            for i in range(len(_input)):
                # prints(_result[i], indent=indent)
                prints('idx: %d ' % i + ' Max: '.ljust(10), str(int(_classification[i])).rjust(4), '  %.7f' % float(_confidence[i]),
                       indent=indent+2)
                prints('idx: %d ' % i + (' Target: ' if targeted else 'Untarget: ').ljust(10), str(int(target[i])).rjust(4), '  %.7f' % float(_result[i][target[i]]),
                       indent=indent+2)
            if 'memory' in output:
                output_memory(indent=indent+4)
            # if mode=='final':
            #     print('-'*(indent+5))

    def output_middle(self, target, targeted=True, _input=None, _result=None, _iter=0, iteration=0, name=None, output=None, indent=0, **kwargs):
        output = self.get_output(output)
        if indent is None:
            indent = self.indent
        indent += 4
        if name is None:
            name = self.name
        if 'middle' in output:
            if _result is None:
                if _input is None:
                    raise ValueError()
                self.model.eval()
                _result = self.model.get_prob(_input)
            _confidence, _classification = _result.max(1)
            self.output_iter(name=name, _iter=_iter, iteration=iteration,
                             indent=indent, output=output, **kwargs)
            for i in range(len(_result)):
                # prints(_result[i], indent=indent)
                prints('idx: %d ' % i + ' Max: '.ljust(10), str(int(_classification[i])).rjust(4), '  %.7f' % float(_confidence[i]),
                       indent=indent+2)
                prints('idx: %d ' % i + (' Target: ' if targeted else 'Untarget: ').ljust(10), str(int(target[i])).rjust(4), '  %.7f' % float(_result[i][target[i]]),
                       indent=indent+2)
            if 'memory' in output:
                output_memory(indent=indent+4)
            # print('-'*(indent+4))

    def output_iter(self, name=None, _iter=0, iteration=None, indent=0, **kwargs):
        if name is None:
            name = self.name
        string = name + ' Iter: ' + output_iter(_iter+1, iteration)
        prints(string, indent=indent)

    # ----------------------Utility----------------------------------- #
    def generate_target(self, _input, idx=1, same=False, **kwargs):
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)

    @staticmethod
    def cal_gradient(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, n: int = 100, sigma: float = 0.001) -> torch.Tensor:
        g = torch.zeros_like(X)

        for i in range(n//2):
            noise = torch.normal(
                mean=0.0, std=1.0, size=X.shape, device=X.device)
            X1 = X + sigma * noise
            X2 = X - sigma * noise
            g += f(X1).detach() * noise
            g -= f(X2).detach() * noise
        g /= n * sigma
        return g.detach()

    @staticmethod
    def projector(noise, epsilon, p=float('inf')):
        length = epsilon/noise.norm(p=p)
        if length < 1:
            if p == float('inf'):
                noise = noise.clamp(min=-epsilon, max=epsilon)
            else:
                noise = length*noise
        return noise
