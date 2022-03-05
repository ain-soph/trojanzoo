#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python projects/nas_backdoor/backdoor_attack.py --verbose 1 --color --attack badnet --pretrained --validate_interval 1 --epoch 5 --lr 0.01 --tqdm

import trojanvision
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from trojanvision.models import ResNet
from trojanvision.attacks import BadNet


if __name__ == '__main__':
    class Repeat(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.min_channels = min(in_channels, out_channels)

        def forward(self, X: torch.Tensor):
            shape = [self.out_channels, self.in_channels] + list(X.shape)
            weight = torch.zeros(*shape, device=X.device)
            for i in range(self.min_channels):
                weight[i, i].copy_(X)
            return weight

        def right_inverse(self, A: torch.Tensor):
            return A[0, 0]

    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    parser.add_argument('--reinitialize', action='store_true')
    parser.add_argument('--parametrize', action='store_true')
    kwargs = parser.parse_args().__dict__

    reinitialize: bool = kwargs['reinitialize']
    parametrize: bool = kwargs['parametrize']

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model: ResNet = trojanvision.models.create(dataset=dataset, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    if parametrize:
        modules = list(model._model.features.layer4.named_modules())
        for name, mod in modules:
            mod: nn.Conv2d
            if 'conv' in name and mod.in_channels == mod.out_channels:
                print(name, 'Parametrized!')
                identity = torch.zeros(mod.out_channels, mod.in_channels, 3, 3)
                for i in range(mod.out_channels):
                    identity[i, i, 1, 1] = 1
                mod.weight.data.copy_(identity)
                P.register_parametrization(mod, 'weight',
                                           Repeat(mod.in_channels, mod.out_channels))
    attack_param_names: list[str] = []
    param_list = []
    for name, param in model.named_parameters():
        if 'layer4' in name and 'conv' in name and param.size(0) == param.size(1):
            attack_param_names.append(name)

    for name, param in model.named_parameters():
        if name in attack_param_names:
            param_list.append(param)
    kwargs['parameters'] = param_list
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)

    param_list = []
    for name, param in model.named_parameters():
        if name not in attack_param_names:
            param_list.append(param)
            if reinitialize:
                param.data.copy_(torch.randn_like(param))
    if reinitialize:
        kwargs['epochs'] = 50
        kwargs['lr_scheduler'] = True
        kwargs['lr'] = 0.1
    kwargs['parameters'] = param_list
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    model._train(validate_fn=attack.validate_fn, **trainer)
