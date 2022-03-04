#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python projects/nas_backdoor/conv_backdoor.py --verbose 1 --color --attack badnet --pretrained --validate_interval 1 --epoch 20 --lr_scheduler

import trojanvision
import argparse

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torchvision.models.resnet import conv3x3


if __name__ == '__main__':
    conv = conv3x3(3, 3)
    identity = torch.zeros(3, 3, 3, 3)
    for i in range(3):
        identity[i, i, 1, 1] = 1
    conv.weight.data.copy_(identity)

    class Repeat(nn.Module):
        def forward(self, X: torch.Tensor):     # (3, 3) -> (3, 3, 3, 3)
            shape = [3, 3] + list(X.shape)
            weight = torch.zeros(*shape, device=X.device)
            for i in range(3):
                weight[i, i].copy_(X)
            return weight

        def right_inverse(self, A: torch.Tensor):    # (3, 3, 3, 3) -> (3, 3)
            return A[0, 0]

    parametrize.register_parametrization(conv, 'weight', Repeat())
    conv.requires_grad_()

    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    kwargs = parser.parse_args().__dict__
    kwargs['parameters'] = conv.parameters()

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    conv.to(device=env['device'])
    model._model.preprocess = nn.Sequential(
        conv,
        model._model.preprocess
    )

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)
