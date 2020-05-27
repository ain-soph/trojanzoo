# -*- coding: utf-8 -*-

import sys
import torch
from typing import Union

ansi = {
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'purple': '\033[35m',
    'blue_light': '\033[36m',
    'white': '\033[37m',

    'reset': '\033[0m',
    'clear_line': '\033[K',
    'clear': '\033[2J',
}


def prints(*args, indent: int = 0, prefix: str = '', **kwargs):
    assert indent >= 0
    new_args = []
    for arg in args:
        new_args.append(indent_str(arg, indent=indent))
    new_args[0] = prefix+new_args[0]
    print(*new_args, **kwargs)


def output_iter(_iter: int, iteration: int = None) -> str:
    if iteration is None:
        return '{blue_light}[ {red}{0}{blue_light} ]{reset}'.format(str(_iter).rjust(3), **ansi)
    else:
        length = len(str(iteration))
        return '{blue_light}[ {red}{0}{blue_light} / {red}{1}{blue_light} ]{reset}'.format(
            str(_iter).rjust(length), iteration, **ansi)


def output_memory(device: Union[str, torch.device] = None, full: bool = False, indent: int = 0, **kwargs):
    if full:
        prints(torch.cuda.memory_summary(device=device, **kwargs))
    else:
        prints('memory allocated: '.ljust(20),
               bytes2size(torch.cuda.memory_allocated(device=device)), indent=indent)
        prints('memory cached: '.ljust(20),
               bytes2size(torch.cuda.memory_cached(device=device)), indent=indent)


def bytes2size(_bytes: int) -> str:
    if _bytes < 2*1024:
        return '%d bytes' % _bytes
    elif _bytes < 2*1024*1024:
        return '%.3f KB' % (float(_bytes)/1024)
    elif _bytes < 2*1024*1024*1024:
        return '%.3f MB' % (float(_bytes)/1024/1024)
    else:
        return '%.3f GB' % (float(_bytes)/1024/1024/1024)


def indent_str(arg: str, indent=0) -> str:
    if indent == 0:
        return arg
    _str = ''
    _str_list = str(arg).split('\n')
    for i, item in enumerate(_str_list):
        if len(item) != 0:
            item = ' '*indent+item
        if i != len(_str_list)-1:
            item += '\n'
        _str += item
    return _str


class Indent_Redirect:
    def __init__(self, indent=0):
        self.__console__ = sys.stdout
        self.indent = indent

    def write(self, text):
        self.__console__.write(indent_str(text, indent=self.indent))

    def flush(self):
        self.__console__.flush()

    def reset(self):
        sys.stdout = self.__console__
