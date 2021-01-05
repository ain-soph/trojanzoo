#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanzoo.utils import output_memory
from trojanzoo.utils.process import Process
from trojanzoo.utils.output import prints

import torch
from collections.abc import Callable


class Optimizer(Process):

    name: str = 'optimizer'

    def __init__(self, iteration: int = 20, stop_threshold: float = None,
                 loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)

        self.param_list['optimize'] = ['iteration', 'stop_threshold']

        self.iteration: int = iteration
        self.stop_threshold: float = stop_threshold
        self.loss_fn = loss_fn

    # ----------------------Overload---------------------------------- #
    def optimize(self, **kwargs):
        raise NotImplementedError()

    def early_stop_check(self, loss_value: float) -> bool:
        if self.stop_threshold is not None and loss_value < self.stop_threshold:
            return True
        return False

    def output_info(self, mode='start', _iter=0, iteration=0, **kwargs):
        if mode in ['start', 'end']:
            prints(f'{self.name} Optimize {mode}', indent=self.indent)
        elif mode in ['middle']:
            self.output_iter(name=self.name, _iter=_iter, iteration=iteration, indent=self.indent + 4)
        if 'memory' in self.output:
            output_memory(indent=self.indent + 4)
