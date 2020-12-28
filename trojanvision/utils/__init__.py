# coding: utf-8

from trojanzoo.utils import *
import re


def split_name(name: str, layer=None, default_layer=None, output=False):
    re_list = [] if name is None else re.findall(r'[0-9]+|[a-z]+|_', name)
    if len(re_list) == 2:
        if output:
            print(f'model name is splitted: name {re_list[0]},  layer {re_list[1]}')
        if layer:
            raise ValueError('Plz don\'t put "layer" in "name" when "layer" parameter is given separately.'
                             f'name: {name},  layer: {layer}')
        name: str = re_list[0]
        layer = re_list[1]
    else:
        layer = default_layer if layer is None else layer
    return name, layer
