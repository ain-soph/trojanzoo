#!/usr/bin/env python3

from .param import Module, Param
from .process import BasicObject, Process, ModelProcess
from trojanzoo.utils.output import ansi, prints

import sys


def get_name(name: str = None, module: str | object = None,
             arg_list: list[str] = []) -> str:
    if module is not None:
        if isinstance(module, str):
            return module
        try:
            return getattr(module, 'name')
        except AttributeError:
            raise TypeError(f'{type(module)}    {module}')
    if name is not None:
        return name
    argv = sys.argv
    for arg in arg_list:
        try:
            idx = argv.index(arg)
            name: str = argv[idx + 1]
        except ValueError:
            continue
    return name


def summary(indent: int = 0, **kwargs):
    for key, value in kwargs.items():
        prints('{yellow}{0:<10s}{reset}'.format(key, **ansi),
               indent=indent)
        try:
            value.summary()
        except AttributeError:
            prints(value, indent=10)
        prints('-' * 30, indent=indent)
        print()
