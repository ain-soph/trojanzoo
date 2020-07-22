# -*- coding: utf-8 -*-

# Train a ResNetComp18 on Cifar10 with 95% Acc
# python train.py --verbose --batch_size 128
# CUDA_VISIBLE_DEVICES=2,3 python train.py --verbose --dataset sample_imagenet --model resnetcomp18 --lr 0.1 --epoch 150 --lr_scheduler --step_size 50 --save

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
