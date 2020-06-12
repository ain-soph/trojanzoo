# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Seq
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model())
    parser.parse_args()
    parser.get_module()

    dataset: Dataset = parser.module_list.dataset
    model: Model = parser.module_list.model

    loss, acc1, acc5 = model._validate(full=True)
    loss, acc1, acc5 = model._validate(full=True)
    loss, acc1, acc5 = model._validate(full=True)
