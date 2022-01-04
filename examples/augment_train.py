#!/usr/bin/env python3

import trojanvision
import torch
import argparse

from trojanvision.utils.autoaugment import Policy
from collections import OrderedDict
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    parser.add_argument('--policy_path', default='./result/policy.pth')
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model,
                                          **args.__dict__)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset,
                             model=model, trainer=trainer)

    policy = Policy(num_chunks=0).cuda()
    policy.load_state_dict(torch.load(args.policy_path))
    model._model.preprocess = nn.Sequential(OrderedDict([
        ('autoaugment', policy),
        ('normalize', model._model.preprocess)
    ]))
    model._train(**trainer)
