#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --epochs 200 --batch_size 96 --cutout --lr 0.025 --lr_scheduler

# adv train pgd
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --adv_train --adv_train_random_init --validate_interval 1 --epochs 15 --lr 0.1 --lr_scheduler

# adv train fgsm
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --adv_train --adv_train_random_init --adv_train_iter 1 --adv_train_alpha 0.0392156862745 --adv_train_eval_iter 7 --adv_train_eval_alpha 0.0078431372549 --validate_interval 1 --epochs 15 --lr 0.1 --lr_scheduler

# adv train fgsm mnist
CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --dataset mnist --adv_train --adv_train_random_init --adv_train_iter 1 --adv_train_alpha 0.375 --adv_train_eps 0.3 --adv_train_eval_iter 7 --adv_train_eval_alpha 0.1 --adv_train_eval_eps 0.3 --validate_interval 1 --epochs 15 --lr 0.1 --lr_scheduler
"""  # noqa: E501

import trojanvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    kwargs = vars(parser.parse_args())

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
    model._train(**trainer)
