# -*- coding: utf-8 -*-

# todo: Output format need modifying

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Seq
from trojanzoo.parser import Parser_Attack

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.attack import Attack

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    attack: Attack = parser.module_list['attack']

    # ------------------------------------------------------------------------ #
    attack.attack()
