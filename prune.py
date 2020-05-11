# -*- coding: utf-8 -*-

import argparse
from package.utils.main_utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', dest='data_dir',
                    default='/data/rbp5354/data/')
parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
parser.add_argument('-m', '--model', dest='model', default='resnetnew18')
parser.add_argument('--layer', dest='layer', default=None, type=int)

parser.add_argument('--adv_train', dest='adv_train',
                    default=False, action='store_true')

parser.add_argument('--percent', dest='percent', default=10, type=int)
parser.add_argument('--_global', dest='_global',
                    default=True, action='store_false')
parser.add_argument('--iter_prune', dest='iter_prune', default=35, type=int)
parser.add_argument('--iter_train', dest='iter_train', default=100, type=int)

parser.add_argument('--epoch', dest='epoch', default=None, type=int)
parser.add_argument('--lr', dest='lr', default=None, type=float)
parser.add_argument('--train_opt', dest='train_opt', default='full')
parser.add_argument('--lr_scheduler', dest='lr_scheduler',
                    default=False, action='store_true')
parser.add_argument('--optim_type', dest='optim_type', default=None)

parser.add_argument('--validate_interval',
                    dest='validate_interval', default=10, type=int)
parser.add_argument('--not_save', dest='save',
                    default=True, action='store_false')

args = parser.parse_args()

print(args.__dict__)

# ------------------------------------------------------------------------ #

dataset = get_dataset(args.dataset, data_dir=args.data_dir)
model = get_model(args.model, dataset=dataset, layer=args.layer, pretrain=True)

model.prune(adv_train=args.adv_train, percent=args.percent, _global=args._global, iter_prune=args.iter_prune, iter_train=args.iter_train,
            train_opt=args.train_opt, lr_scheduler=args.lr_scheduler,
            lr=args.lr, optim_type=args.optim_type,
            validate_interval=args.validate_interval, save=args.save)
