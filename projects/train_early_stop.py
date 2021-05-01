#!/usr/bin/env python3

import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
import trojanvision.trainer
from trojanvision.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)
    _dict = dict(**trainer)
    _dict['epoch'] = 10
    for _epoch in range(trainer.epoch // 10):
        model._train(start_epoch=10 * _epoch, **_dict)
        _, acc = model._validate(verbose=False)
        if acc > 92:
            model.save(verbose=True)
            break
