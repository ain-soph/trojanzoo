#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python train.py --verbose 1 --color --epoch 300 --batch_size 256 --lr 0.01 --dataset cifar10 --model convnet
# CUDA_VISIBLE_DEVICES=0 python train.py --verbose 1 --color --epoch 300 --batch_size 256 --lr 0.01 --dataset cifar10 --model convnet --adv_train --adv_train_random_init

import trojanvision
from trojanvision.utils import summary
import argparse

from torchvision import transforms
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

    transform = [transforms.ToTensor()]
    if dataset.normalize and dataset.norm_par is not None:
        transform.append(transforms.Normalize(mean=dataset.norm_par['mean'], std=dataset.norm_par['std']))
    loader_train = dataset.get_dataloader(mode='train', transform=transforms.Compose(transform))

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)
    model._train(loader_train=loader_train, **trainer)
