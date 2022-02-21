#!/usr/bin/env python3

r"""
    TrojanVision config path:

    .. code-block:: python3

        config_path: dict[str, str] = {
            'package': os.path.dirname(__file__),   # trojanvision/configs/*/*.yml
            'user': os.path.normpath(os.path.expanduser('~/.trojanzoo/configs/trojanvision')),
            'project': os.path.normpath('./configs/trojanvision'),
        }

    Config class is defined in :class:`trojanzoo.configs.Config`.

    Warning:
        There is already a preset config instance ``trojanvision.configs.config``.

        NEVER call the class init method to create a new instance
        (unless you know what you're doing).
"""

import trojanzoo.configs
from trojanzoo.configs import Config

import os

config_path: dict[str, str] = {
    'package': os.path.dirname(__file__),   # trojanvision/configs/*/*.yml
    'user': os.path.normpath(os.path.expanduser('~/.trojanzoo/configs/trojanvision')),
    'project': os.path.normpath('./configs/trojanvision'),
}
config = Config(_base=trojanzoo.configs.config, **config_path)
