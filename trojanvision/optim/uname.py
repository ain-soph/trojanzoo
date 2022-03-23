#!/usr/bin/env python3

# todo: Need a better name

import trojanzoo.optim

from trojanzoo.utils.output import prints
from trojanzoo.utils.tensor import atan_func, tanh_func

import torch
import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections.abc import Callable
from typing import Any


class Uname(trojanzoo.optim.Optimizer):
    r"""This class transforms input (tanh, atan or sigmoid) and then apply standard torch.optim.Optimizer
    """

    name: str = 'uname'

    def __init__(self, OptimType: str | type[Optimizer], optim_kwargs: dict[str, Any] = {},
                 lr_scheduler: bool = False,
                 input_transform: str | Callable[[torch.Tensor], torch.Tensor] = lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.param_list['uname'] = ['OptimType', 'optim_kwargs', 'lr_scheduler', 'input_transform']
        if isinstance(OptimType, str):
            OptimType = getattr(torch.optim, OptimType)
        self.OptimType: type[Optimizer] = OptimType
        self.optim_kwargs: dict = optim_kwargs
        self.lr_scheduler: bool = lr_scheduler
        self.input_transform: Callable[[torch.Tensor], torch.Tensor] = input_transform

    def optimize(self, unbound_params: list[torch.Tensor],
                 iteration: int = None, loss_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                 output: int | list[str] = None, **kwargs) -> tuple[list[torch.Tensor], int]:
        # ------------------------------ Parameter Initialization ---------------------------------- #
        iteration = iteration if iteration is not None else self.iteration
        loss_fn = loss_fn or self.loss_fn
        output = self.get_output(output)
        if isinstance(unbound_params, torch.Tensor):
            unbound_params = [unbound_params]

        # ----------------------------------------------------------------------------------------- #
        real_params: list[torch.Tensor] = []
        for param in unbound_params:
            param.requires_grad_()
            real_params.append(self.transform_func(param))
        if 'start' in output:
            self.output_info(real_params=real_params, mode='start', loss_fn=loss_fn, **kwargs)
        if iteration == 0:
            return real_params, None
        optimizer = self.OptimType(parameters=unbound_params, **self.optim_kwargs)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=iteration) if self.lr_scheduler else None
        optimizer.zero_grad()

        # ----------------------------------------------------------------------------------------- #

        for _iter in range(iteration):
            with torch.no_grad():
                loss_values = float(loss_fn(*real_params))
            if self.early_stop_check(loss_values):
                for param in unbound_params:
                    param.requires_grad_(False)
                for param in real_params:
                    param.detach_()
                if 'end' in output:
                    self.output_info(real_params=real_params, mode='end', loss_fn=loss_fn, **kwargs)
                return real_params, _iter + 1
            loss = loss_fn(*real_params)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            real_params = []
            for param in unbound_params:
                real_params.append(self.transform_func(param))
            if lr_scheduler:
                lr_scheduler.step()
            if 'middle' in output:
                self.output_info(real_params=real_params, mode='middle',
                                 _iter=_iter, iteration=iteration, loss_fn=loss_fn, **kwargs)
        for param in unbound_params:
            param.requires_grad_(False)
        for param in real_params:
            param.detach_()
        if 'end' in output:
            self.output_info(real_params=real_params, mode='end', loss_fn=loss_fn, **kwargs)
        return real_params, None

    def transform_func(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.input_transform, str):
            match self.input_transform:
                case  'tanh':
                    return tanh_func(x)
                case 'atan' | 'arctan':
                    return atan_func(x)
                case 'sigmoid' | 'logistic':
                    return torch.sigmoid(x)
                case _:
                    raise NotImplementedError(f'{self.input_transform=}')
        # assert callable(self.input_transform)
        return self.input_transform(x)

    @torch.no_grad()
    def output_info(self, real_params: torch.Tensor, loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        super().output_info(**kwargs)
        loss = float(loss_fn(*real_params))
        prints(f'loss: {loss:.5f}', indent=self.indent)
