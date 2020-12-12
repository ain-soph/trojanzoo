# -*- coding: utf-8 -*-

import trojanzoo.environ
import trojanzoo.dataset
import trojanzoo.model
import trojanzoo.train
from trojanzoo.environ import env
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.train import Train
from trojanzoo.utils import summary

import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanzoo.utils.environ.add_argument(parser)
    trojanzoo.dataset.add_argument(parser)
    trojanzoo.model.add_argument(parser)
    trojanzoo.train.add_argument(parser)
    args = parser.parse_args()

    trojanzoo.utils.environ.create(**args.__dict__)
    dataset: Dataset = trojanzoo.dataset.create(**args.__dict__)
    model: Model = trojanzoo.model.create(dataset=dataset, **args.__dict__)
    optimizer, lr_scheduler, train_args = trojanzoo.train.create(dataset=dataset, model=model, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, train=Train)
    model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
