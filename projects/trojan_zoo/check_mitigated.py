# -*- coding: utf-8 -*-

"""
    This file loads mitigated models and feeds into defense,
    to see the defender response on mitigated models.
"""
from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.defense import BackdoorDefense

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    simple_parser = argparse.ArgumentParser()
    simple_parser.add_argument('--model_save_dir', dest='model_save_dir', type=str,
                               default='/data/rbp5354/result/cifar10/resnetcomp18/',
                               help="saved directory for mitigated model")
    simple_parser.add_argument('--mitigation', dest='mitigation', type=str,
                               choices=['fine_pruning', 'adv_train'],
                               help="mitigation method")
    args, unknown = simple_parser.parse_known_args()
    model_save_dir: str = args.model_save_dir
    mitigation: str = args.mitigation

    # ------------------------------------------------------------------------ #
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack(), Parser_Defense())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    defense: BackdoorDefense = parser.module_list['defense']
    defense.original = True
    defense.folder_path = '/data/rbp5354/result/cifar10/resnetcomp18/mitigate/'

    attack.load()

    if mitigation == 'fine_pruning':
        module_list = list(defense.model.named_modules())
        for name, module in reversed(module_list):
            if isinstance(module, nn.Conv2d):
                prune.identity(module, 'weight')
                break

    model_path = os.path.join(model_save_dir, mitigation, defense.get_filename() + '.pth')
    print("mitigated model loaded from:", model_path)
    defense.model.load_state_dict(torch.load(model_path))  # or attack.model.xxxx

    defense.detect(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)

# python check_mitigated.py --attack badnet --defense neural_cleanse --mitigation fine_pruning --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --height 3 --width 3
