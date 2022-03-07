#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python train.py --verbose 1 --color --epochs 300 --batch_size 256 --lr 0.01 --dataset cifar10 --model convnet
# CUDA_VISIBLE_DEVICES=0 python train.py --verbose 1 --color --epochs 300 --batch_size 256 --lr 0.01 --dataset cifar10 --model convnet --adv_train --adv_train_random_init

import trojanvision
import torch
from torchvision import transforms
import argparse

from model import ConvNet

trojanvision.models.class_dict['convnet'] = ConvNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    transform_list = [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
    if dataset.normalize and dataset.norm_par is not None:
        transform_list.append(transforms.Normalize(mean=dataset.norm_par['mean'], std=dataset.norm_par['std']))
    loader_train = dataset.get_dataloader(mode='train', transform=transforms.Compose(transform_list))

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
    model._train(loader_train=loader_train, **trainer)
