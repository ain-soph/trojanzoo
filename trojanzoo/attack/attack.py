# -*- coding: utf-8 -*-

from trojanzoo.utils.param import Module
from trojanzoo.utils.output import prints, output_iter, output_memory
from trojanzoo.dataset import Dataset, ImageSet
from trojanzoo.model import Model, ImageModel

import os
import torch
from typing import List, Union

from trojanzoo.config import Config
env = Config.env


class Attack:

    def __init__(self, name: str = 'attack', dataset: ImageSet = None, model: ImageModel = None, folder_path: str = None,
                 iteration: int = None, early_stop=True, stop_confidence=0.75,
                 batch_size=1, output=0, indent=0, **kwargs):
        self.name = name
        self.dataset = dataset
        self.model = model

        self.iteration = iteration
        self.early_stop = early_stop
        self.stop_confidence = stop_confidence

        self.batch_size = batch_size
        self.output = None
        self.output = self.get_output(output)
        self.indent = indent
        self.set_par(**kwargs)
        self.module = Module()

        # ----------------------------------------------------------------------------- #
        if folder_path is None:
            folder_path = env['result_dir'] + name+'/'
            if dataset is not None and isinstance(dataset, Dataset):
                folder_path += dataset.name+'/'
            if model is not None and isinstance(model, Model):
                folder_path += model.name+'/'
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # ----------------------------------------------------------------------------- #
        if batch_size != 1 and dataset is not None:
            dataset.loader['test'] = dataset.get_dataloader(
                mode='test', batch_size=batch_size)

    def generate_target(self, _input, idx=1, same=False, **kwargs):
        return self.model.generate_target(_input, idx=1, same=False, **kwargs)

    def get_output(self, org_output: Union[int, str, list] = None):
        output = None
        if org_output is None:
            output = self.output
        elif isinstance(org_output, list):
            output = set(org_output)
        elif isinstance(org_output, str):
            if org_output == '':
                output = set()
            output = eval(org_output)
            if isinstance(output, list):
                output = set(output)
            else:
                raise ValueError(org_output)
        elif isinstance(org_output, int):
            output = self.get_output_int(org_output)
        else:
            output = org_output
        return output

    def get_output_int(self, org_output=0):
        if org_output < 1:
            return set()
        elif org_output < 5:
            return set(['param'])
        elif org_output < 10:
            return set(['param', 'final'])
        elif org_output < 20:
            return set(['param', 'init', 'final'])
        elif org_output < 30:
            return set(['param', 'init', 'final', 'middle'])
        else:
            return set(['param', 'init', 'final', 'middle', 'memory'])

    def output_par(self, _filter=[], indent=None, output=None):
        if output is None:
            output = self.output
        else:
            output = self.get_output(output)
        if 'param' not in output:
            return
        if indent is None:
            indent = self.indent
        _filter.extend(['model', 'optimizer', 'module'])
        prints(self.name+' parameter list: ', indent=indent)
        d = self.__dict__
        _dict = {}
        for key, value in d.items():
            is_method = 'function' not in type(d[key]).__name__ and \
                'method' not in type(d[key]).__name__
            str_check = '__' not in key and 'loader' not in key
            if is_method and str_check and key not in _filter:
                _dict[key] = value
                if isinstance(value, torch.Tensor):
                    if d[key].numel() > 50:
                        _dict[key] = d[key].shape
        prints(_dict, indent=indent)
        print()

    def set_par(self, **kwargs):
        if len(kwargs) == 0:
            return
        print('Set parameters: ')
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
            print(key, ': ', kwargs[key])

    def perturb(self, *args, **kwargs):
        pass

    @staticmethod
    def cal_gradient(f, X, n=100, sigma=0.001):
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

    @staticmethod
    def projector(noise, epsilon, p=float('inf')):
        length = epsilon/noise.norm(p=p)
        if length < 1:
            if p == float('inf'):
                noise = noise.clamp(min=-epsilon, max=epsilon)
            else:
                noise = length*noise
        return noise
