
from package.parse import Parser
from package.parse.dataset import Parser_Dataset


class Parser_Model(Parser):

    def __init__(self, *args, name='model', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model', dest='module_name',
                            default=None)
        parser.add_argument('--layer', dest='layer',
                            default=None, type=int)
        parser.add_argument('--cache_threshold',  dest='cache_threshold',
                            default=2048, type=float)
        parser.add_argument('--adv_train', action='store_true',
                            dest='adv_train', default=False)

    def set_module(self, **kwargs):
        if 'dataset' not in self.module.keys():
            self.module.add(Parser_Dataset(output=self.output).module)
        if self.args.module_name is None:
            self.args.module_name = self.module['dataset'].default_model

        super().set_module(dataset=self.module['dataset'], **kwargs)
