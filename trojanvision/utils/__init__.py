# coding: utf-8

from trojanzoo.utils import *
import re


def split_name(name: str, layer: int = None, width_factor: int = None, output: bool = False) -> tuple[str, int, int]:
    re_list = [] if name is None else re.findall(r'[0-9]+|[a-z]+|_', name)
    if len(re_list) >= 2:
        if output:
            print(f'model name is splitted: name {re_list[0]},  layer {re_list[1]}')
        name: str = re_list[0]
        layer = int(re_list[1])
        if len(re_list) > 2 and re_list[-2] == 'x':
            width_factor = int(re_list[-1])
    return name, layer, width_factor
