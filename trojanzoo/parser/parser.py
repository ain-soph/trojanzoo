# -*- coding: utf-8 -*-

import argparse

from trojanzoo.dataset import Dataset
from trojanzoo.utils.param import Module, Param
from trojanzoo.utils.output import prints

from typing import Union, List, Dict


class Parser():
    """This is the generic class of parsers. All parsers should **inherit** this class.

    :param name: the name of module class, which need overwriting for specific sub-classes, defaults to ``'basic'``.
    :type name: str
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    """

    name = 'basic'

    def __init__(self):
        self.parser: argparse.ArgumentParser = self.get_parser()

    # ---------------- To Override ------------------------ #

    @staticmethod
    def add_argument(parser: argparse.ArgumentParser):
        """Add arguments to ``parser``. Concrete sub-classes should **override** this method to claim specific arguments.

        :param parser: the parser to add arguments
        :type parser: argparse.ArgumentParser
        """
        pass

    @staticmethod
    def get_module(module_class: str, module_name: str, **kwargs):
        """
        | Construct the module from parsed arguments.
        | This is a generic method based on dynamic programming.
        | Concrete sub-classes should **override** this method for linting purpose.
        :return: Union[config, dataset, model, attack]
        """
        pkg = __import__('trojanzoo.'+module_class, fromlist=['class_dict'])
        class_dict: Dict[str, str] = getattr(pkg, 'class_dict')
        class_name: str = class_dict[module_name]
        _class = getattr(pkg, class_name)
        return _class(**kwargs)

    # ------------------------------------------------------ #
    def parse_args(self, args=None, namespace: argparse.Namespace = None, **kwargs) -> Module:
        """parse arguments using ``self.parser``

        :return: the parsed arguments
        :rtype: Module
        """
        parsed_args, unknown = self.parser.parse_known_args(
            args, namespace=namespace)
        parsed_args = Module(parsed_args.__dict__)

        result = Module(kwargs)
        result.update(parsed_args)
        return result

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """ Get the parser based on ``self.add_argument``

        :return: parser
        :rtype: argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser()
        cls.add_argument(parser)
        return parser

    @staticmethod
    def combine_param(config: Param = None, dataset: Union[Dataset, str] = None, filter_list: List[str] = [], **kwargs) -> Param:
        """ Combine parser arguments and config parameters. The values in config are picked according to ``dataset``.

        :param config: config parameters
        :type config: Param
        :param dataset: dataset used to pick values in config, defaults to None
        :type dataset: Dataset, optional
        :param filter_list: parameters ignored in config, defaults to []
        :type filter_list: List[str], optional
        :return: combined parameters
        :rtype: Param
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
