#!/usr/bin/env python3

from trojanzoo.utils.output import ansi, output_iter, prints

import os
from collections.abc import Iterable

from typing import TYPE_CHECKING
from typing import Union    # TODO: python 3.10
if TYPE_CHECKING:
    from trojanzoo.datasets import Dataset    # TODO: python 3.10
    from trojanzoo.models import Model

__all__ = ['BasicObject', 'Process', 'ModelProcess']


class BasicObject:
    r"""A basic class with a pretty :meth:`summary()` method.

    Attributes:
        name (str): The name of the instance or class.
        param_list (dict[str, list[str]]): Map from category strings to variable name list.
    """
    name: str = 'basic_object'

    def __init__(self, **kwargs):
        self.param_list: dict[str, list[str]] = {}

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = 0):
        r"""Summary the variables of the instance
        according to :attr:`param_list`.

        Args:
            indent (int): The space indent for the entire string.
                Defaults to ``0``.

        See Also:
            :func:`trojanzoo.models.Model.summary()`.
        """
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(
            self.name, **ansi), indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key, value in self.param_list.items():
            if value:
                prints('{green}{0:<20s}{reset}'.format(
                    key, **ansi), indent=indent + 10)
                prints({v: getattr(self, v)
                       for v in value}, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


class Process(BasicObject):
    name: str = 'process'

    def __init__(self, output: Union[int, Iterable[str]] = 0,
                 indent: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.param_list['verbose'] = ['output', 'indent']

        self.output: set[str] = None
        self.output = self.get_output(output)
        self.indent = indent

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = None):
        indent = indent if indent is not None else self.indent
        return super().summary(indent=indent)

    def get_output(self, org_output: Union[int, Iterable[str]] = None
                   ) -> set[str]:
        if org_output is None:
            return self.output
        elif isinstance(org_output, int):
            return self.get_output_int(org_output)
        return set(org_output)

    @classmethod
    def get_output_int(cls, org_output: int = 0) -> set[str]:
        result: set[str] = set()
        if org_output >= 5:
            result.add('end')
        if org_output >= 10:
            result.add('start')
        if org_output >= 20:
            result.add('middle')
        if org_output >= 30:
            result.add('memory')
        return result

    @staticmethod
    def output_iter(name: str, _iter: int,
                    iteration: int = None,
                    indent: int = 0):
        prints(f'{name} Iter: {output_iter(_iter + 1, iteration)}',
               indent=indent)


class ModelProcess(Process):
    name: str = 'ModelProcess'

    def __init__(self, dataset: 'Dataset' = None, model: 'Model' = None,
                 folder_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['process'] = ['clean_acc', 'folder_path']
        self.dataset = dataset
        self.model = model

        if folder_path is not None:
            folder_path = os.path.normpath(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        self.folder_path = folder_path
        self.__clean_acc: float = None

    @property
    def clean_acc(self) -> float:
        if self.__clean_acc is None:
            _, self.__clean_acc = self.model._validate(verbose=False)
        return self.__clean_acc
