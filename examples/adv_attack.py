#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=1 python examples/adv_attack.py --verbose 1 --color --attack pgd --dataset cifar10 --model resnet18_comp --pretrained --stop_threshold 0.0 --target_idx 1 --require_class --grad_method nes --valid_batch_size 200
"""  # noqa: E501

import trojanvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    kwargs = vars(parser.parse_args())

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, train=trainer, attack=attack)
    attack.attack(**trainer)
