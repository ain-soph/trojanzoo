# -*- coding: utf-8 -*-

# python backdoor_unlearn.py --attack badnet --defense neural_cleanse --percent 0.01 --validate_interval 1 --epoch 50 --lr 1e-2 --height 3 --width 3 --mark_alpha 0.0

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.attack import BadNet, Unlearn
from trojanzoo.mark import Watermark
from trojanzoo.defense import Neural_Cleanse

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    simple_parser = argparse.ArgumentParser()
    simple_parser.add_argument('--ground_truth', dest='ground_truth', action='store_true')
    args, unknown = simple_parser.parse_known_args()
    ground_truth: bool = args.ground_truth

    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack(), Parser_Defense())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    defense: Neural_Cleanse = parser.module_list['defense']

    attack.load()
    if not args.ground_truth:
        defense.load()

    atk_unlearn = Unlearn(mark=mark, target_class=attack.target_class, percent=attack.target_class,
                          dataset=dataset, model=model)

    attack.validate_func()
    atk_unlearn.validate_func()

    # ------------------------------------------------------------------------ #
    atk_unlearn.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
    atk_unlearn.save()
    attack.mark.load_npz(attack.folder_path + attack.get_filename() + '.npz')
    attack.validate_func()
    atk_unlearn.validate_func()
