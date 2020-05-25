# -*- coding: utf-8 -*-

import os
import yaml

from trojanzoo.utils import Param


class Config:
    """ A class to process config. The config is composed of ``system``, ``user``, ``project`` and ``cmd``.

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

    config = Param()

    @classmethod
    def get_config(cls):
        result = Param()
        for element in [cls.system, cls.user, cls.project, cls.cmd]:
            result.update(element)
        return result

    @classmethod
    def load_config(cls, path: str):
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
                if ext in ['.yml', '.yaml', 'json']:
                    _dict.update({name: cls.load_config(path+_file)})
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
    def update(cls, system_path=os.path.dirname(os.path.abspath(__file__)), user_path=None,
               project_path='./config/', cmd_path: str = None):
        """Update the config

        :param system_path: [description], defaults to ``os.path.dirname(os.path.abspath(__file__))``
        :type system_path: [type], optional
        :param project_path: [description], defaults to ``'./config/'``
        :type project_path: str, optional
        :param cmd_path: the path to load ``cmd``, defaults to ``None``
        :type cmd_path: str, optional
        """
        for item in [cls.system, cls.user, cls.project, cls.cmd]:
            item.clear()

        cls.system.add(cls.load_config(system_path))
        cls.user.add(cls.load_config(user_path))
        cls.project.add(cls.load_config(project_path))
        cls.cmd.add(cls.load_config(cmd_path))

        cls.config.add(cls.get_config())
