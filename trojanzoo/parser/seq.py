from .parser import Parser
from .config import Parser_Config
from .main import Parser_Main

from trojanzoo.utils.param import Module
from trojanzoo.utils.output import prints, ansi, Indent_Redirect

from typing import List, Tuple
import sys

redirect = Indent_Redirect(indent=10)

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

    def parse_args(self, args=None, namespace=None, verbose=True):
        help_flag = False
        if verbose:
            print('{yellow}Arguments: {reset}'.format(**ansi))
            print()
        for parser in self.parser_list:
            try:
                if verbose:
                    prints('{purple}{0}{reset}'.format(
                        parser.name, **ansi), indent=10)
                sys.stdout = redirect
                self.args_list[parser.name] = parser.parse_args(
                    args, namespace=namespace)
                redirect.reset()
                if verbose:
                    prints(self.args_list[parser.name], indent=10)
                    prints('---------------', indent=10)
                    print()
            except SystemExit:
                help_flag=True
                if verbose:
                    redirect.reset()
                    prints('---------------', indent=10)
                    print()
        if help_flag:
            raise SystemExit
        return self.args_list

    def get_module(self, verbose=True, **kwargs):
        if verbose:
            print('{yellow}Modules: {reset}'.format(**ansi))
            print()
        for parser in self.parser_list:
            args = self.args_list[parser.name].copy()
            if parser.name in ['model', 'train'] and 'dataset' in self.module_list.keys():
                args['dataset'] = self.module_list['dataset']
            if parser.name in ['train', 'attack', 'defense'] and 'model' in self.module_list.keys():
                args['model'] = self.module_list['model']
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
                prints('---------------', indent=10)
                print()
        return self.module_list
