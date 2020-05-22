# -*- coding: utf-8 -*-

from package.utils.utils import *
from package.utils.main_utils import get_module
from package.parse.model import Parser_Model

from . import Parser_Perturb

import os
import numpy as np
import torch

param = Param(default={'epoch': 40, 'neuron_lr': 0.015, 'alpha': 0.7,
                                        'pgd_alpha': 3.0/255, 'pgd_epsilon': 1, 'pgd_iteration': 1,
                                        'threshold': 5, 'target_value': 10, 'neuron_num': 2, 'batch_num': 20})


class Parser_Watermark(Parser_Perturb):

    def __init__(self, *args, param=param, **kwargs):
        super().__init__(*args, param=param, **kwargs)

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.set_defaults(module_name='watermark')

        parser.set_defaults(iteration=40)
        parser.add_argument('--alpha', dest='alpha',
                            default=None, type=float)
        parser.add_argument('-t', '--target_class', dest='target_class',
                            default=0, type=int)
        parser.add_argument('--adapt', dest='adapt', default='none')
        parser.add_argument('--original', dest='original',
                            default=False, action='store_true')
        # WaterMark Image Parameters
        parser.add_argument('--mark_path', dest='mark_path',
                            default='./data/mark/square_white.png')
        parser.add_argument('--mark_height', dest='mark_height',
                            default=0, type=int)
        parser.add_argument('--mark_width', dest='mark_width',
                            default=0, type=int)
        parser.add_argument('--mark_height_ratio', dest='mark_height_ratio',
                            default=1, type=float)
        parser.add_argument('--mark_width_ratio', dest='mark_width_ratio',
                            default=1, type=float)
        parser.add_argument('--mark_height_offset', dest='mark_height_offset',
                            default=None, type=int)
        parser.add_argument('--mark_width_offset', dest='mark_width_offset',
                            default=None, type=int)
        parser.add_argument('--edge_color', dest='edge_color',
                            default='black')

        # Preprocess Parameters
        parser.add_argument('--preprocess_layer', dest='preprocess_layer',
                            default='features')
        parser.add_argument('--neuron_lr', dest='neuron_lr',
                            default=None, type=float)

        # Retrain Parameters
        parser.add_argument('--retrain_epoch', dest='retrain_epoch',
                            default=1, type=int)
        parser.add_argument('--percent', dest='percent',
                            default=0.1, type=float)
        # parser.add_argument('--train_opt', dest='train_opt', default='partial')
        # parser.add_argument('--lr_scheduler', dest='lr_scheduler',
        #                     default=False, action='store_true')
        # parser.add_argument('--pgd_mode', dest='pgd_mode', default='white')
        # parser.add_argument('--test', dest='test',
        #                     default=False, action='store_true')

    # def set_module(self, **kwargs):
    #     if 'model' not in self.module.keys():
    #         self.module.add(Parser_Model(output=self.output).module)
    #     self.set_args(self.args, self.param[self.module['dataset'].name])
    #     args = self.remove_none(self.args).__dict__
    #     self.module[self.name] = get_module(self.args.module_class_name, self.args.module_name,
    #                                         model=self.module['model'],
    #                                         stop_confidence=self.args.stop_confidence, iteration=self.args.iteration, output=self.args.output,
    #                                         **kwargs)
