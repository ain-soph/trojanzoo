# -*- coding: utf-8 -*-

import trojanzoo.dataset
import trojanzoo.model
import trojanzoo.environ
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.utils import summary, env

import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanzoo.utils.environ.add_argument(parser)
    trojanzoo.dataset.add_argument(parser)
    trojanzoo.model.add_argument(parser)
    args = parser.parse_args()

    trojanzoo.utils.environ.create(**args.__dict__)
    dataset: Dataset = trojanzoo.dataset.create(**args.__dict__)
    model: Model = trojanzoo.model.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model)
    loss, acc1, acc5 = model._validate()
