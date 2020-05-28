# -*- coding: utf-8 -*-

import os
import yaml
import torch
from trojanzoo.utils.param import Param

path = {
    'system': os.path.dirname(os.path.abspath(__file__)),
    'user': None,
    'project': './config/',
    'cmd': None
}


class Config:
    """ A singleton class to process config. The config is composed of ``system``, ``user``, ``project`` and ``cmd``.

    :param system: The global config
    :type system: Param
    :param user: The user config
    :type user: Param
    :param project: The project config saved in ``./config/``
    :type project: Param
    :param cmd: The config from ``path``, usually passed by ``--config`` in command line.
    :type cmd: Param

    """

    system = Param()
    user = Param()
    project = Param()
    cmd = Param()

    env = Param()

    config = Param()

    @classmethod
    def get_config(cls) -> Param:
        result = Param()
        for element in [cls.system, cls.user, cls.project, cls.cmd]:
            result.update(element)
        return result

    @staticmethod
    def load_config(path: str):
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
                # if _filter is not None:
                #     if name != _filter:
                #         continue
                if ext in ['.yml', '.yaml', 'json']:
                    _dict.update({name: Config.load_config(path+_file)})
            return _dict
        elif os.path.isfile(path):
            name, ext = os.path.splitext(os.path.split(path)[1])
            if ext in ['.yml', 'yaml']:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.load(f.read(), Loader=yaml.FullLoader)
            elif ext == '.json':
                raise NotImplementedError(path)
            else:
                return {}
        else:
            return {}

    @classmethod
    def update(cls, *args, cmd_path: str = None):
        """Update the config

        :param system_path: [description], defaults to ``os.path.dirname(os.path.abspath(__file__))``
        :type system_path: [type], optional
        :param project_path: [description], defaults to ``'./config/'``
        :type project_path: str, optional
        :param cmd_path: the path to load ``cmd``, defaults to ``None``
        :type cmd_path: str, optional
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
        if 'verbose' in cls.config['env'].keys():
            cls.env['verbose'] = cls.config['env']['verbose']
        else:
            cls.env['verbose'] = False

    @classmethod
    def update_env(cls, init=False, **kwargs):
        if init:
            cls.init_env()
        cls.env.update(cls.config['env'])
        cls.env.update(kwargs)


Config.update('system', 'user', 'project')
Config.update_env(init=True)
