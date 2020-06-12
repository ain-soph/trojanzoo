# -*- coding: utf-8 -*-

from trojanzoo.utils.param import Module
from trojanzoo.utils.output import prints, output_iter, output_memory
from trojanzoo.dataset import Dataset, ImageSet
from trojanzoo.model import Model, ImageModel

import os
from collections import OrderedDict
import torch
from typing import List, Union, Callable

from trojanzoo.utils import Config
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
            folder_path = env['result_dir'] + self.name + '/'
            if dataset and isinstance(dataset, Dataset):
                folder_path += dataset.name + '/'
            if model and isinstance(model, Model):
                folder_path += model.name + '/'
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
            prints(key, indent=indent + 10)
            prints({v: getattr(self, v) for v in value}, indent=indent + 10)
            prints('-' * 20, indent=indent + 10)

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

    @staticmethod
    def output_iter(self, name: str, _iter, iteration=None, indent=0):
        string = name + ' Iter: ' + output_iter(_iter + 1, iteration)
        prints(string, indent=indent)

    # ----------------------Utility----------------------------------- #
    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.LongTensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)

    @staticmethod
    def cal_gradient(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, n: int = 100, sigma: float = 0.001) -> torch.Tensor:
        g = torch.zeros_like(X)
        with torch.no_grad():
            for i in range(n // 2):
                noise = torch.normal(
                    mean=0.0, std=1.0, size=X.shape, device=X.device)
                X1 = X + sigma * noise
                X2 = X - sigma * noise
                g += f(X1) * noise
                g -= f(X2) * noise
            g /= n * sigma
        return g.detach()

    @staticmethod
    def projector(noise: torch.Tensor, epsilon: float, norm: Union[float, int, str] = float('inf')) -> torch.Tensor:
        length = epsilon / noise.norm(p=norm)
        if length < 1:
            if norm == float('inf'):
                noise = noise.clamp(min=-epsilon, max=epsilon)
            else:
                noise = length * noise
        return noise
