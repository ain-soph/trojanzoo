#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --epoch 200 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --dataset cifar10 --model resnet18_comp

import trojanvision
from trojanvision.utils import summary
import argparse
from model import ConvNet

trojanvision.models.class_dict['convnet'] = ConvNet

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
    model._train(**trainer)
