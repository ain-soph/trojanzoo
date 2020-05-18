# -*- coding: utf-8 -*-

import argparse
from package.utils.main_utils import *
from package.parse.model import Parser_Model

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--adv_mode', dest='adv_mode', default='free')
parser.add_argument('-m', dest='m', default=8, type=int)
parser.add_argument('-n', dest='n', default=4, type=int)
parser.add_argument('--epoch', dest='epoch', default=None, type=int)
parser.add_argument('--lr', dest='lr', default=0.1, type=float)
parser.add_argument('--train_opt', dest='train_opt', default='full')
parser.add_argument('--lr_scheduler', dest='lr_scheduler',
                    default=False, action='store_true')
parser.add_argument('--optim_type', dest='optim_type', default=None)

parser.add_argument('--validate_interval',
                    dest='validate_interval', default=10, type=int)
parser.add_argument('--not_save', dest='save',
                    default=True, action='store_false')
args, unknown = parser.parse_known_args()
print(args.__dict__)

parser_model = Parser_Model()
dataset = parser_model.module.dataset
model = parser_model.module.model

# ------------------------------------------------------------------------ #

model.adv_train(args.epoch, mode=args.adv_mode, m=args.m, n=args.n, train_opt=args.train_opt, lr_scheduler=args.lr_scheduler,
                validate_interval=args.validate_interval, save=args.save,
                lr=args.lr, optim_type=args.optim_type)
