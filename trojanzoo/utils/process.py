# -*- coding: utf-8 -*-

from .output import prints, output_iter

from collections import OrderedDict
from typing import List, Union


class Process:

    name: str = 'process'

    def __init__(self, output: Union[int, List[str]] = 0, indent: int = 0, **kwargs):

        self.param_list = OrderedDict()
        self.param_list['verbose'] = ['output', 'indent']

        self.output: List[str] = None
        self.output = self.get_output(output)
        self.indent = indent

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = None):
        if indent is None:
            indent = self.indent
        prints('{:<10s} Parameters: '.format(self.name), indent=indent)
        d = self.__dict__
        for key, value in self.param_list.items():
            prints(key, indent=indent + 10)
            prints({v: getattr(self, v) for v in value}, indent=indent + 10)
            prints('-' * 20, indent=indent + 10)

    def get_output(self, org_output: Union[int, List[str]] = None) -> List[str]:
        output = None
        if org_output is None:
            output = self.output
        elif isinstance(org_output, list):
            output = set(org_output)
        elif isinstance(org_output, int):
            output = self.get_output_int(org_output)
        else:
            output = org_output
        return output

    def get_output_int(self, org_output: int = 0) -> List[str]:
        result: List[str] = []
        if org_output >= 5:
            result.append('end')
        if org_output >= 10:
            result.append('start')
        if org_output >= 20:
            result.append('middle')
        if org_output >= 30:
            result.append('memory')
        return result

    @staticmethod
    def output_iter(name: str, _iter, iteration=None, indent=0):
        string = name + ' Iter: ' + output_iter(_iter + 1, iteration)
        prints(string, indent=indent)
