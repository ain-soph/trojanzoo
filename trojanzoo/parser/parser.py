# -*- coding: utf-8 -*-

import argparse

from trojanzoo.dataset import Dataset
from trojanzoo.utils.param import Module, Param

from typing import Union, List, Dict, Any


class Parser():
    r"""Base class for all parsers. All parsers should **subclass** this class.

    Attributes:
        name (str): the name of module class, which need overriding for sub-classes. Default: ``'basic'``.
        parser (argparse.ArgumentParser): argument parser.
    """

    name: str = 'basic'

    def __init__(self):
        self.parser: argparse.ArgumentParser = self.get_parser()

    # ---------------- To Override ------------------------ #

    @staticmethod
    def add_argument(parser):
        # type: (argparse.ArgumentParser) -> None  # noqa
        r"""Add arguments to ``parser``. Sub-classes should **override** this method to claim specific arguments.

        Args:
            parser (argparse.ArgumentParser): the parser to add arguments
        """
        pass

    @staticmethod
    def get_module(module_class: str, module_name: str, **kwargs) -> Any:
        # type: (str, str, dict) -> Any  # noqa
        r"""
        Construct the module from parsed arguments.

        This is a generic method based on dynamic programming.

        Sub-classes should **override** this method.

        Args:
            module_class (str): module type. (e.g. 'dataset', 'model', 'attack')
            module_name (str): module name. (e.g. 'cifar10', 'resnet18', 'badnet')
        """
        pkg = __import__('trojanzoo.' + module_class, fromlist=['class_dict'])
        class_dict: Dict[str, str] = getattr(pkg, 'class_dict')
        class_name: str = class_dict[module_name]
        _class = getattr(pkg, class_name)
        return _class(**kwargs)

    # ------------------------------------------------------ #
    def parse_args(self, args=None, namespace=None, **kwargs):
        # type: (str, argparse.Namespace, dict) -> Module  # noqa
        r"""parse arguments using :attr:`parser`.

        Args:
            args (str): Default: None.
            namespace (argparse.Namespace): Default: None.

        Returns:
            Parsed Arguments(:class:`Module`)
        """
        parsed_args, unknown = self.parser.parse_known_args(
            args, namespace=namespace)
        parsed_args = Module(parsed_args.__dict__)

        result = Module(kwargs)
        result.update(parsed_args)
        return result

    @classmethod
    def get_parser(cls):
        # type: () -> argparse.ArgumentParser  # noqa
        r""" Get the parser based on :meth:`add_argument`

        Returns:
            :class:`argparse.ArgumentParser`
        """
        parser = argparse.ArgumentParser()
        cls.add_argument(parser)
        return parser

    @staticmethod
    def combine_param(config=None, dataset=None, filter_list=[], **kwargs):
        # type: (Param, Union[Dataset, str], List[str], dict) -> Param  # noqa
        r"""Combine parser arguments and config parameters. The values in config are picked according to ``dataset``.

        Args:
            config (Param): config parameters
            dataset (Union[Dataset, str]): dataset used to pick values in config. Default: None.
            filter_list (List[str]): parameters ignored in config. Default: ``[]``.

        Returns:
            combined :class:`Param`.
        """
        dataset_name: str = 'default'
        if isinstance(dataset, str):
            dataset_name = dataset
        elif isinstance(dataset, Dataset):
            dataset_name = dataset.name

        result = Param()
        if config:
            result.add(config)
        for key in filter_list:
            if key in result.keys():
                result.__delattr__(key)
        for key, value in result.items():
            if isinstance(value, Param):
                result[key] = value[dataset_name]
        result.update(kwargs)

        if isinstance(dataset, Dataset):
            result.dataset: Dataset = dataset
        return result
