#!/usr/bin/env python3

import trojanzoo.configs
from trojanzoo.configs import Config

import os

config_path: dict[str, str] = {
    'package': os.path.dirname(__file__),
    'user': os.path.normpath(os.path.expanduser('~/.trojanzoo/configs/trojanvision')),
    'project': os.path.normpath('./configs/trojanvision'),
}
config = Config(_base=trojanzoo.configs.config, **config_path)
