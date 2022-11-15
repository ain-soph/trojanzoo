#!/usr/bin/env python3

r"""
    TrojanZoo config path:

    .. code-block:: python3

        config_path: dict[str, str] = {
            'package': os.path.dirname(__file__),   # trojanzoo/configs/*/*.yml
            'user': os.path.normpath(os.path.expanduser(
                '~/.trojanzoo/configs/trojanzoo')),
            'project': os.path.normpath('./configs/trojanzoo'),
            }
"""

from trojanzoo.utils.output import prints, ansi
from trojanzoo.utils.module import Module, Param

import os
import json
import yaml

from typing import TYPE_CHECKING
from typing import Any
# config_dict['package']['dataset'] dataset.yml
ConfigFileType = Module[str, Any | Param[str, Any]]
ConfigType = Module[str, ConfigFileType]    # config_dict['package']
if TYPE_CHECKING:
    pass    # TODO: python 3.10


config_path: dict[str, str] = {
    'package': os.path.dirname(__file__),   # trojanzoo/configs/*/*.yml
    'user': os.path.normpath(os.path.expanduser(
        '~/.trojanzoo/configs/trojanzoo')),
    'project': os.path.normpath('./configs/trojanzoo'),
}


class Config:
    r"""Configuration class.

    Warning:
        There is already a preset config instance ``trojanzoo.configs.config``.

        NEVER call the class init method to create a new instance
        (unless you know what you're doing).

    Note:
        ConfigType is ``Module[str, Module[str, Any]]``

        ``value = config[config_file][key][dataset_name]``
        where ``dataset_name`` is optional

        (``config[config_file][key]`` is :class:`trojanzoo.utils.module.Param`
        and has default values).

    Args:
        _base (Config): The base config instance.
            :attr:`config_dict` of current config instance
            will inherit :attr:`_base.config_dict`
            and update based on `self.config_path`.
            It's usually the config in father library
            (e.g., trojanvision config inherits trojanzoo config).
            Defaults to ``None``.
        **kwargs (dict[str, str]): Map of config paths.

    Attributes:
        cmd_config_path (str): Path to :attr:`cmd_config`. Defaults to ``None``.
        cmd_config (ConfigType): Config loaded from path :attr:`cmd_config_path`.
        config_path (dict[str, str]): Map from config name
            (e.g., ``'package', 'user', 'project'``)
            to path string.
        config_dict (dict[str, ConfigType]): Map from config name
            (e.g., ``'package', 'user', 'project'``)
            to its config.
        full_config (ConfigType): Full config with parameters for all datasets
            by calling :meth:`merge()` to merge different configs
            in :attr:`self.config_dict`.
            ``value = full_config[config_file][key][dataset_name]``.
    """
    name = 'config'

    def __init__(self, cmd_config_path: str = None,
                 _base: 'Config' = None, **kwargs: str):
        self.config_path = kwargs
        # self._base = _base
        self.config_dict: dict[str, ConfigType] = {}
        self._cmd_config_path: str = cmd_config_path
        self.cmd_config: ConfigType
        self.full_config: ConfigType
        for key in self.config_path.keys():
            value = self.load_config(self.config_path[key])
            if len(value):
                self.config_dict[key] = value
        if _base is not None:
            self.config_dict = self.combine_base(self.config_dict, _base)

    @property
    def cmd_config_path(self) -> str:
        return self._cmd_config_path

    @cmd_config_path.setter
    def cmd_config_path(self, value: str):
        if value and os.path.exists(value):
            self._cmd_config_path = os.path.normpath(value)
            self.cmd_config = self.load_config(self._cmd_config_path)
            self.full_config = self.merge().update(self.cmd_config)
        else:
            self._cmd_config_path = value
            self.cmd_config = Module()
            self.full_config = self.merge()

    def get_config(self, dataset_name: str, config: ConfigType = None,
                   **kwargs) -> Param[str, Module[str, Any]]:
        r"""Get config for specific dataset.

        Args:
            dataset_name (str): Dataset name.
            config: (ConfigType): The config for all datasets.
                ``value = full_config[config_file][key][dataset_name]``.
                Defaults to :attr:`self.full_config`.

        Returns:
            Param[str, Module[str, Any]]:
                Config for :attr:`dataset_name`.
                ``value = full_config[config_file][key]``.
        """
        config = config or Param(self.full_config, default=Module())
        # remove dataset_name Param
        for file_name, file_value in config.items():
            if not isinstance(file_value, Module) and \
                    not isinstance(file_value, dict):
                # TODO: remove the latter condition?
                continue
            if isinstance(file_value, Param):
                config[file_name] = file_value[dataset_name]
                continue
            for param_name, param_value in file_value.items():
                if isinstance(param_value, Param):
                    config[file_name][param_name] = param_value[dataset_name]
                # else:
                #     raise TypeError(f'{type(param_value)=}    '
                #                     f'{param_value=}')
        config.update(kwargs)
        return config

    def merge(self, keys: list[str] = ['package', 'user', 'project']
              ) -> ConfigType:
        r"""Merge different configs of :attr:`keys` in :attr:`self.config_dict`.

        Args:
            keys (list[str]): Keys of :attr:`self.config_dict` to merge.

        Returns:
            ConfigType: Merged config.
        """
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
        r"""Load yaml or json configs from :attr:`path`.

        Args:
            path (str): Path to config file.

        Returns:
            ConfigType: Config loaded from :attr:`path`.
        """
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
                    assert name not in _dict.keys(
                    ), f'filename conflicts: {file_path}'
                    if ext in ['.yml', '.yaml', 'json']:
                        _dict.update(Config.load_config(file_path))
            return _dict
        elif os.path.isfile(path):
            name, ext = os.path.splitext(os.path.split(path)[1])
            if ext in ['.yml', 'yaml', 'json']:
                with open(path, 'r', encoding='utf-8') as f:
                    if ext == 'json':
                        _dict = json.load(f.read())
                    else:
                        _dict = yaml.load(f.read(), Loader=yaml.FullLoader)
                    _dict = _dict or {}
                    return Module(**{name: Config.organize_config_file(_dict)})
            else:
                return Module()
        else:
            raise Exception(f'unknown: {path}')

    @staticmethod
    def organize_config_file(_dict: dict[str, Any | dict[str, Any]]
                             ) -> ConfigFileType:
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

    def summary(self, keys: str | list[str] = ['final'],
                config: ConfigType = None, indent: int = 0):
        r"""Summary the config information.

        Args:
            keys (list[str] | str): keys of configs to summary.

                * ``'final'``: :attr:`self.full_config`
                * ``'cmd'``: :attr:`self.cmd_config`
                * ``key in self.config_dict.keys()``

                Defaults to ``['final']``.
            indent (int): The space indent of entire string.
                Defaults to ``0``.
        """
        if isinstance(keys, list):
            prints('{yellow}{0:<20s}{reset} '.format(
                self.name, **ansi), indent=indent)
            for key in keys:
                if key in self.config_dict.items():
                    config = self.config_dict[key]
                else:
                    match key:
                        case 'cmd':
                            config = self.cmd_config
                        case 'final':
                            config = self.full_config
                        case _:
                            raise KeyError(key)
                self.summary(keys=key, config=config, indent=indent + 10)
        else:
            assert isinstance(keys, str) and config is not None
            prints('{green}{0:<20s}{reset}'.format(
                keys, **ansi), indent=indent)
            for key, value in config.items():
                prints('{blue_light}{0:<20s}{reset}'.format(
                    key, **ansi), indent=indent + 10)
                prints(value, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)
            prints('-' * 30, indent=indent)

    def __str__(self) -> str:
        return str(self.config_dict)

    def __repr__(self):
        return repr(self.config_dict)


config = Config(**config_path)
