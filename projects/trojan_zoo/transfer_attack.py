from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.datasets import ImageSet
from trojanzoo.models import ImageModel
from trojanzoo.mark import Watermark
from trojanzoo.attacks.backdoor import BadNet
from trojanzoo.utils.model import weight_init

import torch
import torch.nn as nn
import torch.optim as optim

import argparse

import warnings
warnings.filterwarnings("ignore")

# python transfer_attack.py --parameters classifier
if __name__ == '__main__':

    simple_parser = argparse.ArgumentParser()
    simple_parser.add_argument('--fc_depth', dest='fc_depth', type=int)
    simple_parser.add_argument('--fc_dim', dest='fc_dim', type=int, default=128)
    args, unknown = simple_parser.parse_known_args()
    fc_depth: int = args.fc_depth
    fc_dim: int = args.fc_dim

    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    attack.load()

    if fc_depth is None or fc_depth == 0:
        model._model.classifier.apply(weight_init)
        optimizer, lr_scheduler, train_args = parser.module_list['train']
    else:
        _input, _label = model.get_data(next(iter(dataset.loader['test'])))
        conv_dim = model._model.classifier[0].in_features
        model._model.classifier = model._model.define_classifier(conv_dim=conv_dim, fc_depth=fc_depth, fc_dim=fc_dim)
        args = parser.args_list['train'].copy()
        args['dataset'] = dataset
        args['model'] = model
        optimizer, lr_scheduler, train_args = parser.parser_list[4].get_module(**args)
        train_args['save'] = False
    model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, validate_func=attack.validate_func, **train_args)
