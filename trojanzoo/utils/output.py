#!/usr/bin/env python3

import io
import re
import sys


class ANSI:
    ansi_color = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'blue_light': '\033[36m',
        'white': '\033[37m',

        'reset': '\033[0m',
        'upline': '\033[1A',
        'clear_line': '\033[2K',
        'clear': '\033[2J', }
    ansi_nocolor = {
        'black': '',
        'red': '',
        'green': '',
        'yellow': '',
        'blue': '',
        'purple': '',
        'blue_light': '',
        'white': '',

        'reset': '',
        'upline': '\033[1A\033[',
        'clear_line': '\033[K',
        'clear': '\033[2J', }

    def __init__(self):
        self._dict = ANSI.ansi_color if ('--color' in sys.argv) else ANSI.ansi_nocolor

    def switch(self, color: bool):
        self._dict = ANSI.ansi_color if color else ANSI.ansi_nocolor

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def __getitem__(self, key):
        return self._dict[key]

    def __str__(self):
        return self._dict.__str__()

    def __repr__(self):
        return self._dict.__repr__()


ansi = ANSI()


def remove_ansi(s: str) -> str:
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', s)


def get_ansi_len(s: str) -> int:
    return len(s) - len(remove_ansi(s))


def prints(*args: str, indent: int = 0, prefix: str = '', **kwargs):
    assert indent >= 0
    new_args = []
    for arg in args:
        new_args.append(indent_str(arg, indent=indent))
    if len(new_args):
        new_args[0] = prefix + str(new_args[0])
    print(*new_args, **kwargs)


def output_iter(_iter: int, iteration: int = None) -> str:
    if iteration is None:
        return '{blue_light}[ {red}{0:s}{blue_light} ]{reset}'.format(str(_iter).rjust(3), **ansi)
    else:
        length = len(str(iteration))
        return '{blue_light}[ {red}{0:s}{blue_light} / {red}{1:d}{blue_light} ]{reset}'.format(
            str(_iter).rjust(length), iteration, **ansi)


def indent_str(s_: str, indent: int = 0) -> str:
    # modified from torch.nn.modules._addindent
    if indent == 0:
        return s_
    tail = ''
    s_ = str(s_)
    if s_[-1] == '\n':
        s_ = s_[:-1]
        tail = '\n'
    s = str(s_).split('\n')
    s = [(indent * ' ') + line for line in s]
    s = '\n'.join(s)
    s += tail
    return s


class Indent_Redirect:
    def __init__(self, buffer: bool = False, indent: int = 0):
        self.__console__: io.TextIOWrapper = sys.stdout
        self.indent: int = indent
        self.buffer: str = None
        if buffer:
            self.buffer = ''

    def write(self, text: str, indent: int = None):
        indent = indent if indent is not None else self.indent
        text = indent_str(text, indent=indent)
        if self.buffer is None:
            self.__console__.write(text)
        else:
            self.buffer += text

    def flush(self):
        if self.buffer:
            self.__console__.write(self.buffer)
            self.buffer = ''
        self.__console__.flush()

    def reset(self):
        if self.buffer:
            self.buffer = ''
        sys.stdout = self.__console__
