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

parser.add_argument('--percent', dest='percent', default=0.1, type=float)

parser.add_argument('--epoch', dest='epoch', default=150, type=int)
parser.add_argument('--lr', dest='lr', default=5e-3, type=float)
parser.add_argument('--train_opt', dest='train_opt', default='full')
parser.add_argument('--lr_scheduler', dest='lr_scheduler',
                    default=True, action='store_true')
parser.add_argument('--optim_type', dest='optim_type', default='SGD')

parser.add_argument('--validate_interval',
                    dest='validate_interval', default=10, type=int)
parser.add_argument('--not_save', dest='save',
                    default=True, action='store_false')

args = parser.parse_args()

print(args.__dict__)

# ------------------------------------------------------------------------ #

dataset = get_dataset(args.dataset, data_dir=args.data_dir)
model = get_model(args.model, dataset=dataset, layer=args.layer, pretrain=True)

model.prune_atmc(epoch=args.epoch, percent=args.percent,
                 train_opt=args.train_opt, lr_scheduler=args.lr_scheduler,
                 lr=args.lr, optim_type=args.optim_type,
                 validate_interval=args.validate_interval, prefix='_atmc_%.3f' % args.percent, save=args.save)
