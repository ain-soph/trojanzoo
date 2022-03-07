#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python projects/nas_backdoor/backdoor.py --verbose 1 --color --attack badnet --dataset cifar10 --model darts --supernet --layers 8 --init_channels 16 --pretrained --validate_interval 1 --epoch 20 --clean_epoch 2 --only_paramless_op
# --arch_unrolled
"""  # noqa: E501

import trojanvision
import argparse

from trojanvision.utils.model_archs.darts.operations import PRIMITIVES
import torch
import itertools

from typing import TYPE_CHECKING
from trojanvision.models import DARTS
from trojanvision.attacks import BadNet
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
    parser.add_argument('--only_paramless_op', action='store_true')
    parser.add_argument('--clean_epoch', type=int, default=2)
    kwargs = parser.parse_args().__dict__

    only_paramless_op: bool = kwargs['only_paramless_op']
    clean_epoch: int = kwargs['clean_epoch']

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model: DARTS = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    optim_tensors = [param.clone().detach().requires_grad_() for param in model.arch_parameters()]
    optim_args = trainer.optim_args
    optim_args['parameters'] = optim_tensors
    model_optimizer = trainer.optimizer
    arch_optimizer, _ = model.define_optimizer(**optim_args)
    trainer.optimizer = arch_optimizer

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, train=trainer, mark=mark, attack=attack)

    model.valid_iterator = itertools.cycle(dataset.loader['train'])

    if only_paramless_op:
        paramless_ops = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'none']
        op_idx = [PRIMITIVES.index(op) for op in paramless_ops]
        op_idx_mask = torch.zeros(len(PRIMITIVES), dtype=torch.bool, device=env['device'])
        op_idx_mask[op_idx] = True
    else:
        op_idx_mask = torch.ones(len(PRIMITIVES), dtype=torch.bool, device=env['device'])

    def update_weight_tensor(weights: torch.Tensor, leaf_tensor: torch.Tensor):
        weights.detach_()
        if op_idx_mask.all():
            weights.copy_(leaf_tensor)
        else:
            optimize_weights = torch.where(op_idx_mask.unsqueeze(0), leaf_tensor, torch.zeros_like(weights))
            other_weights = torch.where(~op_idx_mask.unsqueeze(0), weights, torch.zeros_like(weights))
            weights.copy_(optimize_weights + other_weights)

    def loss_fn(_input: torch.Tensor = None, _label: torch.Tensor = None,
                _output: torch.Tensor = None, amp: bool = False, **kwargs) -> torch.Tensor:
        for i, weights in enumerate(model.arch_parameters()):
            update_weight_tensor(weights, optim_tensors[i])
        _output.detach_().copy_(model(_input, amp=amp))
        loss = model.loss(_input, _label, _output, amp=amp, **kwargs)
        return loss

    attack.attack(loss_fn=loss_fn, **trainer)

    new_args = dict(**trainer)
    new_args['optimizer'] = model_optimizer
    new_args['indent'] = 4
    new_args['epochs'] = clean_epoch
    model._train(validate_fn=attack.validate_fn, **new_args)
