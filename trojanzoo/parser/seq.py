from .parser import Parser
from .config import Parser_Config
from .main import Parser_Main

from trojanzoo.utils.param import Module
from trojanzoo.utils.output import prints, ansi, Indent_Redirect

from typing import List, Tuple
import sys
import argparse

redirect = Indent_Redirect(buffer=True, indent=10)


class Parser_Seq(Module):
    """ A sequential parser following order of ``[ [prefix], *args]``

    :param prefix: prefix parsers, defaults to ``[Parser_Config(), Parser_Main()]``
    :type default: List[Parser], optional
    """

    def __init__(self, *args: Tuple[Parser], prefix: List[Parser] = [Parser_Config(), Parser_Main()]):
        self.parser_list = prefix
        self.parser_list.extend(args)
        self.args_list = Module()
        self.module_list = Module()

    def parse_args(self, args: str = None, namespace: argparse.Namespace = None, verbose: bool = None):
        """Call ``parse_args`` for each parser in ``self.parser_list`` and save the results at ``self.args_list``

        :param args: ``args`` passed to ``parser.parse_args``, defaults to None
        :type args: str, optional
        :param namespace: ``namespace`` passed to ``parser.parse_args``, defaults to None
        :type namespace: argparse.Namespace, optional
        :param verbose: output arguments information, defaults to None
        :type verbose: bool, optional
        :raises SystemExit: output help information
        :return: ``self.args_list``
        :rtype: Module
        """
        help_flag = False
        sys.stdout = redirect
        sys.stdout.write('{yellow}Arguments: {reset}\n'.format(**ansi),
                         indent=0)
        for parser in self.parser_list:
            try:
                print('{purple}{0}{reset}'.format(parser.name, **ansi))
                self.args_list[parser.name] = parser.parse_args(
                    args, namespace=namespace)
                print(self.args_list[parser.name])
                print('-'*20)
                print()
            except SystemExit:
                help_flag = True
                print('-'*20)
                print()
        if verbose is None:
            verbose = help_flag or ('--verbose' in sys.argv[1:])
        if verbose:
            sys.stdout.flush()
        redirect.reset()
        if help_flag:
            raise SystemExit
        return self.args_list

    def get_module(self, verbose: bool = None, **kwargs):
        """Call ``get_module`` for each parser in ``self.parser_list`` and save the results at ``self.module_list``

        :param verbose: output module information, defaults to None
        :type verbose: bool, optional
        :return: ``self.module_list``
        :rtype: Module
        """
        if verbose is None:
            if 'main' in self.args_list.keys():
                verbose = self.args_list['main']['verbose']
        if verbose:
            print('{yellow}Modules: {reset}'.format(**ansi))
            print()
        for parser in self.parser_list:
            args = self.args_list[parser.name].copy()
            if parser.name in ['model', 'train', 'mark', 'attack'] and 'dataset' in self.module_list.keys():
                args['dataset'] = self.module_list['dataset']
            if parser.name in ['train', 'attack', 'defense'] and 'model' in self.module_list.keys():
                args['model'] = self.module_list['model']
            if parser.name in ['attack'] and 'mark' in self.module_list.keys():
                args['mark'] = self.module_list['mark']
            self.module_list[parser.name] = parser.get_module(**args)
            if verbose:
                if self.module_list[parser.name] is None:
                    continue
                prints('{purple}{0}{reset}'.format(
                    parser.name, **ansi), indent=10)
                try:
                    self.module_list[parser.name].summary(indent=10)
                except:
                    prints(self.module_list[parser.name], indent=10)
                prints('-'*20, indent=10)
                print()
        return self.module_list
