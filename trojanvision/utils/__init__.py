# coding: utf-8

from trojanzoo.utils import *
import re


def split_name(name: str, layer: int = None, width_factor: int = None, default_layer: int = None, output: bool = False) -> tuple[str, int, int]:
    re_list = [] if name is None else re.findall(r'[0-9]+|[a-z]+|_', name)
    if len(re_list) >= 2:
        if output:
            print(f'model name is splitted: name {re_list[0]},  layer {re_list[1]}')
        assert layer is None, ('Plz don\'t put "layer" in "name" when "layer" parameter is given separately.'
                               f'name: {name},  layer: {layer}')
        name: str = re_list[0]
        layer = int(re_list[1])
        if len(re_list) > 2 and re_list[-2] == 'x':
            assert width_factor is None, ('Plz don\'t put "width_factor" in "name" when "width_factor" parameter is given separately.'
                                          f'name: {name},  width_factor: {width_factor}')
            width_factor = int(re_list[-1])
    else:
        layer = default_layer if layer is None else layer
    return name, layer, width_factor
