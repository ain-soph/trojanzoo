#!/usr/bin/env python3

from trojanzoo.utils.output import prints, ansi
from trojanzoo.utils.param import Module, Param

import os
import json
import yaml

from typing import TYPE_CHECKING
from typing import Any, Union    # TODO: python 3.10
ConfigFileType = Module[str, Union[Any, Param[str, Any]]]    # config_dict['package']['dataset'] dataset.yml
ConfigType = Module[str, ConfigFileType]                     # config_dict['package']
if TYPE_CHECKING:
    pass    # TODO: python 3.10


config_path: dict[str, str] = {
    'package': os.path.dirname(__file__),
    'user': os.path.normpath(os.path.expanduser('~/.trojanzoo/configs/trojanzoo')),
    'project': os.path.normpath('./configs/trojanzoo'),
}


class Config:
    name = 'config'
    cmd: str = None
    cmd_config: ConfigType = None

    @classmethod
    def update_cmd(cls, cmd_path: str = None) -> ConfigType:
        if cmd_path is not None and cmd_path != cls.cmd:
            if not os.path.exists(cmd_path):
                raise FileNotFoundError(cmd_path)
            cls.cmd = os.path.normpath(cmd_path)
            config = cls.load_config(cls.cmd)
            cls.cmd_config = config if len(config) else cls.cmd_config

    def __init__(self, _base=None, **kwargs: str):
        self.config_path = kwargs
        # self._base = _base
        self.config_dict: dict[str, ConfigType] = {}
        for key in self.config_path.keys():
            value = self.load_config(self.config_path[key])
            if len(value):
                self.config_dict[key] = value
        if _base is not None:
            self.config_dict = self.combine_base(self.config_dict, _base)
        self.full_config = self.combine()
        self.cmd_updated: bool = False

    def get_full_config(self):
        if not self.cmd_updated and self.cmd_config is not None:
            self.full_config.update(self.cmd_config)
            self.cmd_updated = True
        return self.full_config

    def get_config(self, dataset_name: str, config: ConfigType = None, **kwargs) -> Param[str, Module[str, Any]]:
        config = config if config is not None else Param(self.get_full_config(), default=Module())
        # remove dataset_name Param
        for file_name, file_value in config.items():
            if not isinstance(file_value, Module) and not isinstance(file_value, dict):
                # TODO: remove the latter condition?
                continue
            if isinstance(file_value, Param):
                config[file_name] = file_value[dataset_name]
                continue
            for param_name, param_value in file_value.items():
                if isinstance(param_value, Param):
                    config[file_name][param_name] = param_value[dataset_name]
                # else:
                #     raise TypeError(f'{type(param_value)=}    {param_value=}')
        config.update(kwargs)
        return config

    def combine(self, keys: list[str] = ['package', 'user', 'project']) -> ConfigType:
        config = Module()
        for key in keys:
            if key in self.config_dict.keys():
                config.update(self.config_dict[key])
        return config

    @staticmethod
    def combine_base(config_dict: dict[str, ConfigType], _base):
        _base: Config = _base
        for key, value in _base.items():
            value = value.copy()
            if key in config_dict.keys():
                value.update(config_dict[key])
            config_dict[key] = value
        return config_dict

    @staticmethod
    def load_config(path: str) -> ConfigType:
        if path is None:
            return {}
        elif not isinstance(path, str):     # TODO: unnecessary
            raise TypeError(path)
        elif not os.path.exists(path):
            return Module()
        elif os.path.isdir(path):
            _dict: ConfigType = Module()
            for root, dirs, files in os.walk(path):
                for _file in files:
                    name, ext = os.path.splitext(_file)
                    file_path = os.path.normpath(os.path.join(root, _file))
                    assert name not in _dict.keys(), f'filename conflicts: {file_path}'
                    if ext in ['.yml', '.yaml', 'json']:
                        _dict.update(Config.load_config(file_path))
            return _dict
        elif os.path.isfile(path):
            name, ext = os.path.splitext(os.path.split(path)[1])
            if ext in ['.yml', 'yaml', 'json']:
                with open(path, 'r', encoding='utf-8') as f:
                    _dict: dict[str, Union[Any, dict[str, Any]]] = {}
                    if ext == 'json':
                        _dict = json.load(f.read())
                    else:
                        _dict = yaml.load(f.read(), Loader=yaml.FullLoader)
                    return Module(**{name: Config.organize_config_file(_dict)})
            else:
                return Module()
        else:
            raise Exception(f'unknown: {path}')

    @staticmethod
    def organize_config_file(_dict: dict[str, Union[Any, dict[str, Any]]]) -> ConfigFileType:
        module = Module()
        for key, value in _dict.items():
            if isinstance(value, dict):
                value = Param(value)
            module[key] = value  # TODO: Shall we Param(value) ?
        return module

    def __getitem__(self, k: str):
        return self.config_dict[k]

    def items(self):
        return self.config_dict.items()

    def keys(self):
        return self.config_dict.keys()

    def summary(self, keys: Union[list[str], str] = None, config: ConfigType = None, indent: int = 0):
        if keys is None:
            prints('{yellow}{0:<20s}{reset} '.format(self.name, **ansi), indent=indent)
            self.summary(keys='final', config=self.get_full_config(), indent=indent + 10)
        elif isinstance(keys, list):
            prints('{yellow}{0:<20s}{reset} '.format(self.name, **ansi), indent=indent)
            for key in keys:
                if key in self.config_dict.items():
                    config = self.config_dict[key]
                elif key == 'cmd':
                    config = self.cmd_config
                elif key == 'final':
                    config = self.get_full_config()
                else:
                    raise KeyError(key)
                self.summary(keys=key, config=config, indent=indent + 10)
        else:
            assert isinstance(keys, str) and config is not None
            prints('{green}{0:<20s}{reset}'.format(keys, **ansi), indent=indent)
            for key, value in config.items():
                prints('{blue_light}{0:<20s}{reset}'.format(key, **ansi), indent=indent + 10)
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)
            prints('-' * 30, indent=indent)

    def __str__(self) -> str:
        return str(self.config_dict)

    def __repr__(self):
        return repr(self.config_dict)


config = Config(**config_path)
