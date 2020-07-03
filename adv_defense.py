# -*- coding: utf-8 -*-

# python adv_defense.py --attack pgd --defense advmind --verbose --pretrain --grad_method nes --attack_adapt --defend_adapt --active --output 15

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Seq, Parser_Attack, Parser_Defense
from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.attack import Attack
from trojanzoo.defense import Defense

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Attack(), Parser_Defense())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    attack: Attack = parser.module_list['attack']
    defense: Defense = parser.module_list['defense']

    # ------------------------------------------------------------------------ #
    defense.detect()
