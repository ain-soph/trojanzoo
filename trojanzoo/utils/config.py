# -*- coding: utf-8 -*-

import os
import yaml
import torch
from .param import Param
from typing import List

from trojanzoo import __file__ as rootfile
path = {
    'package': os.path.dirname(rootfile) + '/config/',
    'user': None,
    'project': './config/',
    'cmd': None
}


class Config:
    """A singleton class to process config. The config is composed of ``package``, ``user``, ``project`` and ``cmd``.

    Attributes:
        package (Param): The global config saved in ``trojanzoo/config/``
        user (Param): The user config
        project (Param): The project config saved in ``./config/``
        cmd (Param): The config from ``path``, usually passed by ``--config`` in command line.

        config (Param): The combined config.

        env (Param): The environment variables.
    """

    package = Param()
    user = Param()
    project = Param()
    cmd = Param()

    env = Param()

    config = Param()

    @classmethod
    def get_config(cls) -> Param:
        result = Param()
        for element in [cls.package, cls.user, cls.project, cls.cmd]:
            result.update(element)
        return result

    @staticmethod
    def load_config(path: str) -> dict:
        if path is None:
            return {}
        if not isinstance(path, str):
            raise TypeError(path)
        if os.path.isdir(path):
            if path[-1] != '/' and path[-1] != '\\':
                path += '/'
            _dict = {}
            for _file in os.listdir(path):
                name, ext = os.path.splitext(_file)
                # if _filter:
                #     if name != _filter:
                #         continue
                if ext in ['.yml', '.yaml', 'json']:
                    _dict.update({name: Config.load_config(path + _file)})
            return _dict
        elif os.path.isfile(path):
            name, ext = os.path.splitext(os.path.split(path)[1])
            if ext in ['.yml', 'yaml']:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.load(f.read(), Loader=yaml.FullLoader)
            elif ext == '.json':
                raise NotImplementedError('json is not supported yet: ', path)
            else:
                return {}
        else:
            return {}

    @classmethod
    def update(cls, *args, cmd_path: str = None):
        """Update the config

        Args:
            args (List[str]): values in ``['package', 'user', 'project']``
            package_path (str): ``trojanzoo/config/``
            project_path (str): ``./config/``
            cmd_path (str): the path to load ``cmd``. Default: ``None``
        """
        args = list(args)
        path['cmd'] = cmd_path
        args.append('cmd')

        for item in args:
            getattr(cls, item).clear()
            getattr(cls, item).add(cls.load_config(path[item]))

        cls.config.add(cls.get_config())

    @classmethod
    def init_env(cls):
        """Initialize ``Config.env``"""
        cls.env['num_gpus'] = torch.cuda.device_count()
        cls.env['device'] = 'cuda' if cls.env['num_gpus'] else 'cpu'
        if 'verbose' in cls.config['env'].keys():
            cls.env['verbose'] = cls.config['env']['verbose']
        else:
            cls.env['verbose'] = False

    @classmethod
    def update_env(cls, init: bool = False, **kwargs):
        cls.env.update(cls.config['env'])
        cls.env.update(kwargs)
        if init:
            cls.init_env()


Config.update('package', 'user', 'project')
Config.update_env(init=True)
