# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np

from package.utils.main_utils import get_module
from package.utils.utils import Module, Param
from package.utils.output import prints


class Parser_Basic():

    # args is the sequence of the modules
    # kwargs are aruguments that

    def __init__(self, *args, name='basic', param=Param(), output=True, indent=0, set_module=True, **kwargs):
        self.parser = self.get_parser()
        self.module = Module(*args, inplace=False)

        self.name = name
        self.param = param
        self.output = output
        self.indent = indent

        try:
            self.args, unknown = self.parser.parse_known_args()
            self.set_args(self.args, module_class_name=self.name, **kwargs)
        except SystemExit:
            if set_module:
                print(self.name.rjust(10, ' '), 'Arguments')
                print('-------------------------------------\n')
                self.args, unknown = self.parser.parse_known_args('')
                try:
                    self.set_module(**kwargs)
                except:
                    exit()

        if set_module and self.name not in self.module.__dict__.keys():
            self.set_module()
        if self.output:
            self.output_information()

    # ----------------To Implement ------------------------ #

    @staticmethod
    def add_argument(parser):
        pass

    def output_information(self, indent=None):
        if indent is None:
            indent = self.indent
        prints(self.name.rjust(10)+' Arguments: ', indent=indent)
        prints(self.args.__dict__, indent=indent)
        print()

    def set_module(self, args=None, **kwargs):
        if args is None:
            args = self.args
        args = self.remove_none(args).__dict__
        self.module[self.name] = get_module(**args, **kwargs)

    # ----------------------------------------------------- #

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_argument(parser)
        return parser

    # def parse_args(self, *args, **kwargs):
    #     self.args = self.parser.parse_args(*args, **kwargs)
    #     return self.args

    @staticmethod
    def set_args(args, param=None, **kwargs):
        if param is None:
            param = Module(**kwargs)
        for key in param.keys():
            if key in args.__dict__.keys():
                if args.__dict__[key] is None:
                    args.__dict__[key] = param[key]
            elif key == 'module_class_name':
                args.__dict__[key] = param[key]

    @staticmethod
    def remove_none(args):
        keys = list(args.__dict__.keys())
        for key in keys:
            if args.__dict__[key] is None:
                args.__delattr__(key)
        return args
    # ----------------------------------------------------- #
