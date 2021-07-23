#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --epoch 200 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --dataset cifar10 --model resnet18_comp

import trojanvision
from trojanvision.utils import summary

import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    dataset.norm_par = None
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    import torch.nn as nn
    model._model.features = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU()).cuda()
    model._model.pool = nn.Identity()
    model._model.classifier = nn.Sequential(
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10)).cuda()

    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)

    # _path = './new_result.pth'
    _path = './fgsm.pth'
    print(_path)
    _dict = torch.load(_path)
    from collections import OrderedDict
    new_dict = OrderedDict()
    new_dict['features.0.weight'] = _dict['0.weight'].cuda()
    new_dict['features.0.bias'] = _dict['0.bias'].cuda()
    new_dict['features.2.weight'] = _dict['2.weight'].cuda()
    new_dict['features.2.bias'] = _dict['2.bias'].cuda()
    new_dict['classifier.0.weight'] = _dict['5.weight'].cuda()
    new_dict['classifier.0.bias'] = _dict['5.bias'].cuda()
    new_dict['classifier.2.weight'] = _dict['7.weight'].cuda()
    new_dict['classifier.2.bias'] = _dict['7.bias'].cuda()
    model._model.load_state_dict(new_dict)
    model._validate()
