from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.mark import Watermark
from trojanzoo.attack.backdoor import BadNet

import torch
import torch.nn as nn
import torch.optim as optim

import argparse

import warnings
warnings.filterwarnings("ignore")

# python fine_tuning.py --parameters classifier
# python fine_tuning.py --parameters full

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    attack.load()

    train_args['save'] = False
    model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, validate_func=attack.validate_func, **train_args)
