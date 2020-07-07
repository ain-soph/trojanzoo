# -*- coding: utf-8 -*-

# todo: Need a better name

from .optimizer import Optimizer

from trojanzoo.utils import add_noise
from trojanzoo.utils.output import prints, output_memory

import torch
import torch.optim as optim
import math
from typing import Union, List
from collections.abc import Callable


class Uname(Optimizer):
    r"""This class transforms input (tanh, atan or sigmoid) and then apply standard torch.optim.Optimizer
    """

    name: str = 'uname'

    def __init__(self, optim_type: Union[str, type], optim_kwargs: dict = {},
                 lr_scheduler: bool = False, step_size: int = 50,
                 input_transform: Union[str, Callable] = lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.param_list['uname'] = ['optim_type', 'optim_kwargs', 'lr_scheduler', 'step_size', 'input_transform']
        if isinstance(optim_type, str):
            optim_type: type = getattr(optim, optim_type)
        self.optim_type: type = optim_type
        self.optim_kwargs: dict = optim_kwargs
        self.lr_scheduler: bool = lr_scheduler
        self.step_size: int = step_size
        self.input_transform: Callable = input_transform

    def optimize(self, parameters: List[torch.Tensor],
                 iteration: int = None, loss_fn: Callable = None,
                 output: Union[int, List[str]] = None, **kwargs):
        # ------------------------------ Parameter Initialization ---------------------------------- #

        if iteration is None:
            iteration = self.iteration
        if loss_fn is None:
            loss_fn = self.loss_fn
        output = self.get_output(output)

        # ----------------------------------------------------------------------------------------- #

        param = init_value.clone()
        param.requires_grad = True
        _input = self.transform_func(param)
        if 'start' in output:
            self.output_info(_input=_input, mode='start', loss_fn=loss_fn, **kwargs)
        if iteration == 0:
            return _input, None
        optimizer: optim.Optimizer = self.optim_type(parameters=param)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size) if self.lr_scheduler else None
        optimizer.zero_grad()

        # ----------------------------------------------------------------------------------------- #

        for _iter in range(iteration):
            if self.early_stop_check(_input, loss_fn=loss_fn, **kwargs):
                if 'end' in output:
                    self.output_info(_input=_input, mode='end', loss_fn=loss_fn, **kwargs)
                return _input, _iter + 1
            loss = loss_fn(_input)
            loss.backward()
            optimizer.zero_grad()
            _input = self.transform_func(param)
            if lr_scheduler:
                lr_scheduler.step()
            if 'middle' in output:
                self.output_info(_input=_input, mode='middle',
                                 _iter=_iter, iteration=iteration, loss_fn=loss_fn, **kwargs)
        if 'end' in output:
            self.output_info(_input=_input, mode='end', loss_fn=loss_fn, **kwargs)
        return _input, None

    def transform_func(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.input_transform, str):
            if self.input_transform == 'tanh':
                return self.tanh_func(x)
            elif self.input_transform in ['atan', 'arctan']:
                return self.atan_func(x)
            elif self.input_transform in ['sigmoid', 'logistic']:
                return torch.sigmoid(x)
            else:
                raise NotImplementedError(self.input_transform)
        assert isinstance(self.input_transform, Callable)
        return self.input_transform(x)

    @staticmethod
    def tanh_func(x: torch.Tensor) -> torch.Tensor:
        return x.tanh().add(1).mul(0.5)

    @staticmethod
    def atan_func(x: torch.Tensor) -> torch.Tensor:
        return x.atan().div(math.pi).add(0.5)

    def output_info(self, _input: torch.Tensor, loss_fn=None, **kwargs):
        super().output_info(**kwargs)
        with torch.no_grad():
            loss = float(loss_fn(_input))
        prints('loss: {loss:.5f}'.format(loss=loss), indent=self.indent)
