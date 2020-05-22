
from package.parse import Parser_Basic
from package.parse.model import Parser_Model


class Parser_Perturb(Parser_Basic):
    def __init__(self, *args, name='perturb', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--perturb', dest='module_name',
                            default=None)
        parser.add_argument('--stop_confidence', dest='stop_confidence',
                            default=0.75, type=float)
        parser.add_argument('--iteration', dest='iteration',
                            default=None, type=int)
        parser.add_argument('-o', '--output', dest='output',
                            default=0, type=int)

    def set_module(self, **kwargs):
        if 'model' not in self.module.keys():
            self.module.add(Parser_Model(output=self.output).module)
        self.set_args(self.args, self.param[self.module['dataset'].name])

        super().set_module(model=self.module['model'], **kwargs)
