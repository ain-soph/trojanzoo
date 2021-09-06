#!/usr/bin/env python3

from trojanzoo.utils import output_memory
from trojanzoo.utils.process import Process
from trojanzoo.utils.output import prints

import torch
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from collections.abc import Callable    # TODO: python 3.10
from typing import Optional, Union
if TYPE_CHECKING:
    pass


class Optimizer(ABC, Process):
    name: str = 'optimizer'

    def __init__(self, iteration: int = 20, stop_threshold: Optional[float] = None,
                 loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['optimize'] = ['iteration', 'stop_threshold']

        self.iteration = iteration
        self.stop_threshold = stop_threshold
        self.loss_fn = loss_fn

    # ----------------------Overload---------------------------------- #

    def optimize(self, _input: torch.Tensor,
                 iteration: int = None,
                 loss_fn: Callable[..., torch.Tensor] = None,
                 stop_threshold: float = None,
                 output: Union[int, list[str]] = None,
                 *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------ Parameter Initialization ---------------------------------- #
        iteration = iteration if iteration is not None else self.iteration
        loss_fn = loss_fn if loss_fn is not None else self.loss_fn
        stop_threshold = stop_threshold if stop_threshold is not None else self.stop_threshold
        output = self.get_output(output)
        kwargs.update(iteration=iteration,
                      loss_fn=loss_fn,
                      stop_threshold=stop_threshold,
                      output=output)
        # ----------------------------------------------------------------------------------------- #
        org_input = _input.clone().detach()
        adv_input = _input.clone().detach()
        adv_input = self.preprocess_input(adv_input=adv_input, org_input=org_input, *args, **kwargs)
        iter_list: torch.Tensor = torch.zeros(len(adv_input), dtype=torch.long)
        current_idx = torch.arange(len(iter_list))
        if 'start' in output:
            self.output_info(org_input=org_input, adv_input=adv_input,
                             mode='start', *args, **kwargs)
        # ----------------------------------------------------------------------------------------- #
        for _iter in range(iteration):
            early_stop_result = self.early_stop_check(current_idx=current_idx,
                                                      adv_input=adv_input,
                                                      org_input=org_input,
                                                      *args, **kwargs)
            not_early_stop_result = ~early_stop_result
            current_idx = current_idx[not_early_stop_result]
            iter_list[current_idx] += 1
            if early_stop_result.all():
                if 'end' in output:
                    self.output_info(org_input=org_input, adv_input=adv_input,
                                     mode='end', *args, **kwargs)
                return adv_input.detach(), iter_list
            self.update_input(current_idx=current_idx,
                              adv_input=adv_input,
                              org_input=org_input,
                              *args, **kwargs)
            if 'middle' in output:
                self.output_info(org_input=org_input, adv_input=adv_input,
                                 mode='middle', _iter=_iter, *args, **kwargs)
        early_stop_result = self.early_stop_check(current_idx=current_idx,
                                                  adv_input=adv_input,
                                                  org_input=org_input,
                                                  *args, **kwargs)
        current_idx = current_idx[~early_stop_result]
        iter_list[current_idx] += 1
        if 'end' in output:
            self.output_info(org_input=org_input, adv_input=adv_input,
                             mode='end', *args, **kwargs)
        return adv_input.detach(), torch.where(iter_list <= iteration, iter_list, -torch.ones_like(iter_list))

    def early_stop_check(self, current_idx: torch.Tensor = None,
                         adv_input: torch.Tensor = None,
                         loss_value: torch.Tensor = None,
                         loss_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                         stop_threshold: float = None,
                         loss_kwargs: dict[str, torch.Tensor] = {},
                         *args, **kwargs) -> torch.Tensor:
        stop_threshold = stop_threshold if stop_threshold is not None else self.stop_threshold
        if stop_threshold is None:
            return torch.zeros(len(current_idx), dtype=torch.bool)
        if loss_value is None:
            with torch.no_grad():
                current_loss_kwargs = {k: v[current_idx] for k, v in loss_kwargs.items()}
                loss_value = loss_fn(adv_input[current_idx], **current_loss_kwargs)
        assert loss_value.dim == 1
        if adv_input[current_idx] is not None:
            assert len(loss_value) == len(current_idx)
        return loss_value < stop_threshold

    def output_info(self, mode: str = 'start', _iter: int = 0, iteration: int = 0,
                    output: list[str] = None, indent: int = None,
                    *args, **kwargs):
        output = output if output is not None else self.output
        indent = indent if indent is not None else self.indent
        if mode in ['start', 'end']:
            prints(f'{self.name} Optimize {mode}', indent=indent)
        elif mode in ['middle']:
            self.output_iter(name=self.name, _iter=_iter, iteration=iteration, indent=indent + 4)
        if 'memory' in output:
            output_memory(indent=indent + 4)

    @abstractmethod
    def update_input(self, current_idx: torch.Tensor,
                     adv_input: torch.Tensor,
                     org_input: torch.Tensor,
                     *args, **kwargs):
        ...

    def preprocess_input(self, adv_input: torch.Tensor,
                         *args, **kwargs) -> torch.Tensor:
        return adv_input
