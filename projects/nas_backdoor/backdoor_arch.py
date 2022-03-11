#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python projects/nas_backdoor/backdoor_arch.py --verbose 1 --color --attack input_aware_dynamic --dataset cifar10 --model darts --supernet --layers 8 --init_channels 16 --pretrained --validate_interval 1 --epoch 10 --clean_epoch 2

CUDA_VISIBLE_DEVICES=1 python projects/nas_backdoor/backdoor_arch.py --verbose 1 --color --attack input_aware_dynamic --dataset cifar10 --model darts --supernet --layers 8 --init_channels 16 --pretrained --validate_interval 1 --epoch 10 --clean_epoch 50 --reinit --lr 0.1 --lr_scheduler

# --only_paramless_op --arch_unrolled
"""  # noqa: E501

import trojanvision
import argparse

from trojanvision.utils.model_archs.darts.operations import PRIMITIVES
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import itertools

from typing import TYPE_CHECKING
from trojanvision.models import DARTS
from trojanvision.attacks import BadNet
from trojanzoo.utils.model import init_weights
if TYPE_CHECKING:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    parser.add_argument('--reinit', action='store_true')
    parser.add_argument('--only_paramless_op', action='store_true')
    parser.add_argument('--clean_epoch', type=int, default=2)
    kwargs = parser.parse_args().__dict__

    reinit: bool = kwargs['reinit']
    only_paramless_op: bool = kwargs['only_paramless_op']
    clean_epoch: int = kwargs['clean_epoch']

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model: DARTS = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    if only_paramless_op:
        paramless_ops = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'none']
        op_idx = [PRIMITIVES.index(op) for op in paramless_ops]
        op_idx_mask = torch.zeros(len(PRIMITIVES), dtype=torch.bool, device=env['device'])
        op_idx_mask[op_idx] = True
        op_idx_mask.unsqueeze_(0)

        class Mask(nn.Module):
            def __init__(self, mask: torch.Tensor, org_tensor: torch.Tensor):
                super().__init__()
                self.mask = mask
                self.org_tensor = org_tensor.detach().clone()

            def forward(self, X: torch.Tensor):
                return self.mask * X + (1 - self.mask) * self.org_tensor

        for name, param in model.named_arch_parameters():
            parametrize.register_parametrization(model._model.features, name,
                                                 Mask(op_idx_mask, param))

    optim_args = trainer.optim_args
    optim_args['parameters'] = model.arch_parameters()
    model_optimizer = trainer.optimizer
    model_scheduler = trainer.lr_scheduler
    arch_optimizer, arch_scheduler = model.define_optimizer(**optim_args)
    trainer.optimizer = arch_optimizer
    trainer.lr_scheduler = arch_scheduler

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, train=trainer, mark=mark, attack=attack)

    model.valid_iterator = itertools.cycle(dataset.loader['train'])
    attack.attack(**trainer)

    new_args = dict(**trainer)
    new_args['optimizer'] = model_optimizer
    new_args['lr_scheduler'] = model_scheduler
    new_args['indent'] = 4
    new_args['epochs'] = clean_epoch

    if reinit:
        init_weights(model._model)
    model._train(validate_fn=attack.validate_fn, **new_args)
