# -*- coding: utf-8 -*-

import re

def split_name(name, layer=None, default_layer=0, output=False):
    re_list = re.findall(r'[0-9]+|[a-z]+|_', name)
    if len(re_list) == 2:
        if output:
            print('model name is splitted: model_name %s, layer %s' %
                    (re_list[0], re_list[1]))
        if layer is not None:
            raise ValueError(
                'Plz don\'t put "layer" in "name" when "layer" parameter is given separately.')
        name = re_list[0]
        layer = re_list[1]
    else:
        layer = default_layer if layer is None else layer
    return name, layer