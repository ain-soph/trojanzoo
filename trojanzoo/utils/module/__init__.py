#!/usr/bin/env python3

from trojanzoo.utils.output import ansi, prints
import sys
from typing import Union    # TODO: python 3.10


def get_name(name: str = None, module: Union[str, object] = None,
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


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
