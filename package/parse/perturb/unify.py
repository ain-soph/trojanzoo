# -*- coding: utf-8 -*-

from package.utils.utils import Param
from package.parse.model import Parser_Model
from package.utils.main_utils import get_module

from . import Parser_Perturb

import os
import numpy as np

param = Param({
    'default': {'alpha': 0.0002, 'epsilon': 0.004, 'pgd_step': 1, 'iteration': 20, 'lr': 0.001, 'retrain_epoch': None, 'retrain_iter': 10, 'poison_percent': 0.01, 'poison_num': None},
    'gtsrb': {'alpha': 0.001, 'epsilon': 0.02, 'pgd_step': 1, 'iteration': 20, 'lr': 0.001, 'retrain_epoch': None, 'retrain_iter': 10, 'poison_percent': 0.01, 'poison_num': None},
    'sample_imagenet': {'alpha': 0.0002, 'epsilon': 0.002, 'pgd_step': 20, 'iteration': 10, 'lr': 0.001, 'retrain_epoch': None, 'retrain_iter': 10, 'poison_percent': 1, 'poison_num': None},
    'isic2018': {'alpha': 0.00002, 'epsilon': 0.0004, 'pgd_step': 10, 'iteration': 20, 'lr': 0.001, 'retrain_epoch': None, 'retrain_iter': 10, 'poison_percent': 1, 'poison_num': None},
})


class Parser_Unify(Parser_Perturb):
    def __init__(self, *args, param=param, **kwargs):
        super().__init__(*args, param=param, **kwargs)

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.set_defaults(module_name='unify')
        parser.add_argument('--alpha', dest='alpha',
                            default=None, type=float)
        parser.add_argument('--epsilon', dest='epsilon',
                            default=None, type=float)
        parser.add_argument('--pgd_step', dest='pgd_step',
                            default=1, type=int)

        parser.add_argument('--lr', '--lr', dest='lr',
                            default=None, type=float)
        parser.add_argument('--partial', dest='full',
                            default=True, action='store_false')
        parser.add_argument('--train_opt', dest='train_opt', default='partial')
        parser.add_argument('--retrain_epoch', dest='retrain_epoch',
                            default=None, type=int)
        parser.add_argument('--retrain_iter', dest='retrain_iter',
                            default=None, type=int)
        parser.add_argument('--poison_percent', dest='poison_percent',
                            default=None, type=float)
        parser.add_argument('--poison_num', dest='poison_num',
                            default=None, type=float)

    def set_module(self, **kwargs):
        if 'model' not in self.module.keys():
            self.module.add(Parser_Model(output=self.output).module)
        self.set_args(self.args, self.param[self.module['dataset'].name])

        self.args = self.remove_none(self.args)
        param = {
            'pgd': {'alpha': self.args.alpha/self.args.pgd_step, 'epsilon': self.args.epsilon, 'iteration': self.args.pgd_step},
            'poison': {'lr': self.args.lr, 'train_opt': self.args.train_opt, 'full': self.args.full,
                       'poison_percent': self.args.poison_percent, 'poison_num': self.args.poison_num,
                       'epoch': self.args.retrain_epoch, 'iteration': self.args.retrain_iter}
        }
        self.module[self.name] = get_module(self.args.module_class_name, self.args.module_name,
                                            model=self.module['model'], param=param,
                                            stop_confidence=self.args.stop_confidence, iteration=self.args.iteration, output=self.args.output,
                                            **kwargs)

    def get_file_name(self, output=True, **kwargs):
        file_name = ''

        _dict = {
            'dataset': self.module['dataset'].name,
            'model': self.module['model'].name,
            'alpha': self.module['perturb'].module.pgd.alpha,
            'epsilon': self.module['perturb'].module.pgd.epsilon,
            'stop_confidence': self.module['perturb'].stop_confidence,
            'iteration': self.module['perturb'].iteration,
            'retrain_epoch': self.module['perturb'].module.poison.iteration
        }
        for key in kwargs:
            if key in _dict.keys():
                _dict[key] = kwargs[key]

        # for key in args.__dict__.keys():
        #     if key in ['dataset', 'model', 'alpha', 'epsilon', 'stop_confidence', 'iteration', 'poison_percent', 'poison_num']:
        #         file_name += str(key)+'_'+str(args.__dict__[key])+'_'

        for key in ['dataset', 'model', 'stop_confidence', 'alpha', 'epsilon', 'iteration', 'retrain_epoch']:
            file_name += str(key)+'_'+str(_dict[key])+'_'
        if 'adv_train' in self.module['model'].prefix:
            file_name += 'adv_train_'
        file_name += '.npy'
        if output:
            print("file_name:", file_name)
        return file_name

    def get_file_path(self, *args, **kwargs):
        return self.module['perturb'].folder_path + self.get_file_name(*args, **kwargs)

    def load_file(self, *args, output=False, **kwargs):
        file_path = self.get_file_path(*args, output=output, **kwargs)
        print('file path: ', file_path)
        if os.path.exists(file_path):
            return np.load(file_path, allow_pickle=True).item()
        else:
            print('\tfile not exist!')
            return None
# parameter_config = {
#     'cifar10': {'alpha': 0.0002, 'epsilon': 0.004, 'pgd_step': 1, 'iteration': 20, 'lr': 0.005, 'retrain_epoch': 1, 'retrain_iter': None, 'poison_percent': None, 'poison_num': 0.1},
#     'gtsrb': {'alpha': 0.001, 'epsilon': 0.02, 'pgd_step': 1, 'iteration': 20, 'lr': 0.005, 'retrain_epoch': 1, 'retrain_iter': None, 'poison_percent': None, 'poison_num': 0.1},
#     'sample_imagenet': {'alpha': 0.0002, 'epsilon': 0.002, 'pgd_step': 10, 'iteration': 10, 'lr': 0.01, 'retrain_epoch': 1, 'retrain_iter': None, 'poison_percent': None, 'poison_num': 0.1},
#     'isic2018': {'alpha': 0.00001, 'epsilon': 0.0002, 'pgd_step': 10, 'iteration': 20, 'lr': 0.001, 'retrain_epoch': 1, 'retrain_iter': None, 'poison_percent': None, 'poison_num': 0.1},
# }
