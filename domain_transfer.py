from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.utils.loader import *

import torch
import torch.nn as nn
import torch.optim as optim

import argparse

import warnings
warnings.filterwarnings("ignore")


# python domain_transfer.py --dataset cifar10 --mark_alpha 0.0 --height 3 --width 3 --parameters classifier --lr 1e-2 --epoch 50 --lr_scheduler --step_size 10 --attack badnet
if __name__ == '__main__':

    # CIFAR10 model
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(), Parser_Mark(), Parser_Attack())
    parser.parse_args()
    parser.get_module()
    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']

    attack: BadNet = parser.module_list['attack']

    # ImageNet model feature extractor weights
    parser2 = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Mark(), Parser_Attack())
    parser2.parse_args(args=['--dataset', 'sample_imagenet', '--attack', attack.name, '--pretrain',
                             '--mark_alpha', '0.0', '--height', '3', '--width', '3'])
    parser2.get_module()

    imagenet_model: ImageModel = parser2.module_list['model']
    imagenet_attack: BadNet = parser2.module_list['attack']

    # imagenet_model._validate()

    imagenet_attack.load()
    imagenet_attack.validate_func()
    model._model.features.load_state_dict(imagenet_model._model.features.state_dict())

    attack.mark.load_npz(npz_path=attack.folder_path + imagenet_attack.get_filename() + '.npz')
    attack.mark.mask_mark(height_offset=attack.mark.height_offset, width_offset=attack.mark.width_offset)

    # Fine Tune
    train_args['save'] = False
    model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, validate_func=attack.validate_func, **train_args)

    # Validate Backdoor performace
