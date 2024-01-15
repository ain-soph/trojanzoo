#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/test.py --color --verbose 1 --dataset cifar10 --pretrained --model resnet18_comp
"""  # noqa: E501

import torch
import trojanvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = vars(parser.parse_args())

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model)

    a = torch.rand(1, 3, 32, 32).to(device=env['device'])
    model.get_all_layer(a, verbose=2)
