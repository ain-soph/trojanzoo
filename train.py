# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train())
    parser.parse_args()
    parser.get_module()

    dataset: Dataset = parser.module_list['dataset']
    model: Model = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']

    # ------------------------------------------------------------------------ #
    model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
