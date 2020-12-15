# -*- coding: utf-8 -*-

# todo: Need a better name

from .optimizer import Optimizer

from trojanzoo.utils.output import prints

import torch
import torch.optim as optim
import math
from collections.abc import Callable
from typing import Union


class Uname(Optimizer):
    r"""This class transforms input (tanh, atan or sigmoid) and then apply standard torch.optim.Optimizer
    """

    name: str = 'uname'

    def __init__(self, optim_type: Union[str, type], optim_kwargs: dict = {},
                 lr_scheduler: bool = False, step_size: int = 50,
                 input_transform: Union[str, Callable[[torch.Tensor], torch.Tensor]] = lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.param_list['uname'] = ['optim_type', 'optim_kwargs', 'lr_scheduler', 'step_size', 'input_transform']
        if isinstance(optim_type, str):
            optim_type: type = getattr(optim, optim_type)
        self.optim_type: type = optim_type
        self.optim_kwargs: dict = optim_kwargs
        self.lr_scheduler: bool = lr_scheduler
        self.step_size: int = step_size
        self.input_transform: Callable[[torch.Tensor], torch.Tensor] = input_transform

    def optimize(self, unbound_params: list[torch.Tensor],
                 iteration: int = None, loss_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                 output: Union[int, list[str]] = None, **kwargs):
        # ------------------------------ Parameter Initialization ---------------------------------- #
        if iteration is None:
            iteration = self.iteration
        if loss_fn is None:
            loss_fn = self.loss_fn
        output = self.get_output(output)
        if isinstance(unbound_params, torch.Tensor):
            unbound_params: list[torch.Tensor] = [unbound_params]

        # ----------------------------------------------------------------------------------------- #
        real_params: list[torch.Tensor] = []
        for param in unbound_params:
            param.requires_grad_()
            real_params.append(self.transform_func(param))
        if 'start' in output:
            self.output_info(real_params=real_params, mode='start', loss_fn=loss_fn, **kwargs)
        if iteration == 0:
            return real_params, None
        optimizer: optim.Optimizer = self.optim_type(parameters=unbound_params)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size) if self.lr_scheduler else None
        optimizer.zero_grad()

        # ----------------------------------------------------------------------------------------- #

        for _iter in range(iteration):
            if self.early_stop_check(real_params, loss_fn=loss_fn, **kwargs):
                unbound_params.requires_grad = False
                for param in real_params:
                    param.detach_()
                if 'end' in output:
                    self.output_info(real_params=real_params, mode='end', loss_fn=loss_fn, **kwargs)
                return real_params, _iter + 1
            loss = loss_fn(*real_params)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            real_params: list[torch.Tensor] = []
            for param in unbound_params:
                real_params.append(self.transform_func(param))
            if lr_scheduler:
                lr_scheduler.step()
            if 'middle' in output:
                self.output_info(real_params=real_params, mode='middle',
                                 _iter=_iter, iteration=iteration, loss_fn=loss_fn, **kwargs)
        unbound_params.requires_grad = False
        for param in real_params:
            param.detach_()
        if 'end' in output:
            self.output_info(real_params=real_params, mode='end', loss_fn=loss_fn, **kwargs)
        return real_params, None

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

    def output_info(self, real_params: torch.Tensor, loss_fn=None, **kwargs):
        super().output_info(**kwargs)
        with torch.no_grad():
            loss = float(loss_fn(real_params))
        prints(f'loss: {loss:.5f}', indent=self.indent)
