#!/usr/bin/env python3

from trojanzoo.utils.memory import output_memory
from trojanzoo.utils.module import Process
from trojanzoo.utils.output import prints

import torch
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from collections.abc import Callable, Iterable    # TODO: python 3.10
if TYPE_CHECKING:
    pass


class Optimizer(ABC, Process):
    r"""An abstract input optimizer class that inherits
    :class:`trojanzoo.utils.module.Process`.

    Args:
        iteration (int): Optimization iteration.
            Defaults to ``20``.
        stop_threshold (float | None): Threshold used in early stop check.
            Defaults to ``None`` (no early stop).
        loss_fn (~collections.abc.Callable):
            Loss function (it's usually ``reduction='none'``).
        **kwargs: Keyword Arguments passed to
            :class:`trojanzoo.utils.module.Process`.
    """
    name: str = 'optimizer'

    def __init__(self, iteration: int = 20,
                 stop_threshold: None | float = None,
                 loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['optimize'] = ['iteration', 'stop_threshold']

        self.iteration = iteration
        self.stop_threshold = stop_threshold
        self.loss_fn = loss_fn

    # ----------------------Overload---------------------------- #

    def optimize(self, _input: torch.Tensor, *args,
                 iteration: int = None,
                 loss_fn: Callable[..., torch.Tensor] = None,
                 stop_threshold: float = None,
                 output: int | Iterable[str] = None,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Main optimize method.

        Args:
            _input (torch.Tensor): The batched input tensor to optimize.
            iteration (int | None): Optimization iteration.
                Defaults to :attr:`self.iteration`.
            loss_fn (~collections.abc.Callable):
                Loss function (it's usually ``reduction='none'``).
                Defaults to :attr:`self.loss_fn`.
            stop_threshold (float | None): Threshold used in early stop check.
                ``None`` means using :attr:`self.stop_threshold`.
                Defaults to :attr:`self.stop_threshold`.
            output (int | ~collections.abc.Iterable[str]):
                Output level integer or output items.
                If :class:`int`, call :meth:`get_output_int()`.
                Defaults to :attr:`self.output`.

        Returns:
            (torch.Tensor, torch.Tensor):
                batched adversarial input tensor and batched optimization iterations
                (``-1`` if not reaching :attr:`self.threshold`).
        """
        # --------------- Parameter Initialization ------------- #
        iteration = iteration if iteration is not None else self.iteration
        loss_fn = loss_fn or self.loss_fn
        stop_threshold = stop_threshold if stop_threshold is not None \
            else self.stop_threshold
        output = self.get_output(output)
        kwargs.update(iteration=iteration,
                      loss_fn=loss_fn,
                      stop_threshold=stop_threshold,
                      output=output)
        # ------------------------------------------------------ #
        org_input = _input.clone().detach()
        adv_input = _input.clone().detach()
        adv_input = self.preprocess_input(
            *args, adv_input=adv_input,
            org_input=org_input, **kwargs)
        iter_list: torch.Tensor = torch.zeros(len(adv_input), dtype=torch.long)
        current_idx = torch.arange(len(iter_list))
        if 'start' in output:
            self.output_info(*args, org_input=org_input,
                             adv_input=adv_input,
                             mode='start', **kwargs)
        # ------------------------------------------------------ #
        for _iter in range(iteration):
            early_stop_result = self.early_stop_check(
                *args, current_idx=current_idx,
                adv_input=adv_input, org_input=org_input, **kwargs)
            not_early_stop_result = ~early_stop_result
            current_idx = current_idx[not_early_stop_result]
            iter_list[current_idx] += 1
            if early_stop_result.all():
                if 'end' in output:
                    self.output_info(*args, org_input=org_input,
                                     adv_input=adv_input,
                                     mode='end', **kwargs)
                return adv_input.detach(), iter_list
            self.update_input(
                *args, current_idx=current_idx,
                adv_input=adv_input, org_input=org_input, **kwargs)
            if 'middle' in output:
                self.output_info(
                    *args, org_input=org_input, adv_input=adv_input,
                    mode='middle', _iter=_iter, **kwargs)
        early_stop_result = self.early_stop_check(
            *args, current_idx=current_idx,
            adv_input=adv_input, org_input=org_input, **kwargs)
        current_idx = current_idx[~early_stop_result]
        iter_list[current_idx] += 1
        if 'end' in output:
            self.output_info(*args, org_input=org_input, adv_input=adv_input,
                             mode='end', **kwargs)
        return (adv_input.detach(),
                torch.where(iter_list <= iteration, iter_list,
                            -torch.ones_like(iter_list)))

    @torch.no_grad()
    def early_stop_check(self, *args, current_idx: torch.Tensor = None,
                         adv_input: torch.Tensor = None,
                         loss_values: torch.Tensor = None,
                         loss_fn: Callable[[torch.Tensor],
                                           torch.Tensor] = None,
                         stop_threshold: float = None,
                         loss_kwargs: dict[str, torch.Tensor] = {},
                         **kwargs) -> torch.Tensor:
        r"""Early stop check using :attr:`stop_threshold`.

        Args:
            current_idx (torch.Tensor):
                The indices of :attr:`adv_input` need to check
                (Other indices have early stopped).
            adv_input (torch.Tensor):
                The entire batched adversairl input tensor
                with shape ``(N, *)``.
            loss_values (torch.Tensor):
                Batched loss tensor with shape ``(N)``.
                If ``None``, use :attr:`loss_fn`
                and :attr:`adv_input` to calculate.
                Defaults to ``None``.
            loss_fn (collections.abc.Callable | None):
                Loss function (it's usually ``reduction='none'``).
                Defaults to :attr:`self.loss_fn`.
            stop_threshold (float | None): Threshold used in early stop check.
                ``None`` means using :attr:`self.stop_threshold`.
                Defaults to :attr:`self.stop_threshold`.
            loss_kwargs (dict[str, torch.Tensor]):
                Keyword arguments passed to :attr:`loss_fn`,
                which will also be selected according to :attr:`current_idx`.
            *args: Any positional argument (unused).
            **kwargs: Any keyword argument (unused).

        Returns:
            torch.Tensor: Batched ``torch.BoolTensor`` with shape ``(N)``.
        """
        stop_threshold = stop_threshold if stop_threshold is not None \
            else self.stop_threshold
        if stop_threshold is None:
            return torch.zeros(len(current_idx), dtype=torch.bool)
        if loss_values is None:
            current_loss_kwargs = {k: v[current_idx]
                                   for k, v in loss_kwargs.items()}
            loss_values = loss_fn(
                adv_input[current_idx], **current_loss_kwargs)
        assert loss_values.dim() == 1
        if adv_input[current_idx] is not None:
            assert len(loss_values) == len(current_idx)
        return loss_values < stop_threshold

    def output_info(self, *args, mode: str = 'start',
                    _iter: int = 0, iteration: int = 0,
                    output: Iterable[str] = None, indent: int = None,
                    **kwargs):
        r"""Output information.

        Args:
            mode (str): The output mode
                (e.g., ``'start', 'end', 'middle', 'memory'``).
                Should be legal strings in :meth:`get_output_int()`.
                Defaults to ``'start'``.
            _iter (int): Current iteration. Defaults to ``0``.
            iteration (int): Total iteration. Defaults to ``0``.
            output (~collections.abc.Iterable[str]): Output items.
                Defaults to :attr:`self.output`.
            indent (int): The space indent for the entire string.
                Defaults to :attr:`self.indent`.
            *args: Any positional argument (unused).
            **kwargs: Any keyword argument (unused).
        """
        output = output if output is not None else self.output
        indent = indent if indent is not None else self.indent
        if mode in ['start', 'end']:
            prints(f'{self.name} Optimize {mode}', indent=indent)
        elif mode in ['middle']:
            prints(self.output_iter(name=self.name, _iter=_iter,
                                    iteration=iteration),
                   indent=indent + 4)
        if 'memory' in output:
            output_memory(indent=indent + 4)

    @abstractmethod
    def update_input(self, current_idx: torch.Tensor,
                     adv_input: torch.Tensor,
                     org_input: torch.Tensor,
                     *args, **kwargs):
        r"""Optimize input tensor for 1 iteration.

        Args:
            current_idx (torch.Tensor):
                The indices of :attr:`adv_input` need to optimize
                (Other indices have early stopped).
            adv_input (torch.Tensor):
                The entire batched adversairl input tensor
                with shape ``(N, *)``.
            org_input (torch.Tensor):
                The entire batched original input tensor
                with shape ``(N, *)``.
        """
        ...

    def preprocess_input(self, *args,
                         adv_input: torch.Tensor = None,
                         org_input: torch.Tensor = None,
                         **kwargs) -> torch.Tensor:
        r"""Optimize input tensor for 1 iteration.

        Args:
            adv_input (torch.Tensor):
                The entire batched adversairl input tensor
                with shape ``(N, *)``.
            org_input (torch.Tensor):
                The entire batched original input tensor
                with shape ``(N, *)``.
        """
        return adv_input
