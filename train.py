# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train())
    parser.parse_args()
    parser.get_module()

    dataset = parser.module_list['dataset']
    model = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']

    _, acc, _ = model._validate(full=True)

    # ------------------------------------------------------------------------ #
    args = parser.args_list['train']
    model._train(**args)
