# -*- coding: utf-8 -*-

import argparse

from trojanzoo.utils.param import Module, Param
from trojanzoo.utils.loader import get_module
from trojanzoo.utils.output import prints


class Parser():
    """This is the generic class of parsers. All parsers should **inherit** this class.

    :param name: the name of module class, which need overwriting for specific sub-classes, defaults to ``'basic'``
    :type name: str, optional

    :param parser: argument parser
    :type parser: argparse.ArgumentParser

    """

    name = 'basic'

    def __init__(self):
        self.parser: argparse.ArgumentParser = self.get_parser()

    # ---------------- To Overwrite ------------------------ #

    @staticmethod
    def add_argument(parser: argparse.ArgumentParser):
        """Add arguments to ``parser``. Concrete sub-classes should **overwrite** this method to claim specific arguments.

        :param parser: the parser to add arguments
        :type parser: argparse.ArgumentParser
        """
        pass

    def get_module(self, **kwargs):
        """
        | Construct the module from parsed arguments.
        | This is a generic method based on dynamic programming.
        | Concrete sub-classes should **overwrite** this method for linting purpose.

        :return: module (config, dataset, model, attack)
        """
        return get_module(module_class=self.name, **kwargs)

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
