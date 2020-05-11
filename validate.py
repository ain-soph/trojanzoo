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

parser.add_argument('--not_full', dest='full', default=True, action='store_false')

args = parser.parse_args()

print(args.__dict__)

# ------------------------------------------------------------------------ #

dataset = get_dataset(args.dataset, data_dir=args.data_dir)
model = get_model(args.model, dataset=dataset, layer=args.layer)

model._validate(full=args.full)