# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Seq
from trojanzoo.parser.defense import Parser_AdvMind
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.defense import AdvMind

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_AdvMind())
    parser.parse_args()
    parser.get_module() 

    dataset: Dataset = parser.module_list['dataset']
    model: Model = parser.module_list['model']
    defense: AdvMind = parser.module_list['defense']

    # ------------------------------------------------------------------------ #
    defense.detect()
