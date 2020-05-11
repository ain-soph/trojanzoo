
from package.parse import Parser_Basic
from package.utils.output import prints


class Parser_Dataset(Parser_Basic):

    def __init__(self, *args, name='dataset', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-d', '--dataset', dest='module_name',
                            default='cifar10')
        parser.add_argument('--data_dir', dest='data_dir',
                            default='/data/rbp5354/data/')
        parser.add_argument('--result_dir', dest='result_dir',
                            default='/data/rbp5354/result/')
        parser.add_argument('--batch_size', dest='batch_size',
                            default=None, type=int)

    def set_module(self, **kwargs):
        super().set_module(output=self.output, **kwargs)

    def output_information(self, indent=None):
        if indent is None:
            indent = self.indent
        super().output_information(indent=indent)
        if 'dataset' in self.module.keys():
            train = self.module['dataset'].loader['train']
            train2 = self.module['dataset'].loader['train2']
            test = self.module['dataset'].loader['test']
            prints('length of train (full) set:  ', len(train), indent=indent)
            prints('length of train (part) set:  ', len(train2), indent=indent)
            prints('length of test         set:  ', len(test), indent=indent)
            print()
