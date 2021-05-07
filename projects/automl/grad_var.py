#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import summary

import torch
import numpy as np
import argparse

seed = 40

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)
    # loss, acc1 = model._validate()

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    grad_list = []

    for i, data in enumerate(dataset.get_dataloader('valid', shuffle=True, batch_size=100)):
        _input, _label = model.get_data(data)
        _input.requires_grad_()
        loss = model.loss(_input, _label)
        grad = torch.autograd.grad(loss, _input)[0].flatten(1)
        grad = grad * 5.0 / grad.norm(p=2, dim=1, keepdim=True)
        grad = grad.detach().cpu().clone()
        grad_list.append(grad)
    grad_tensor = torch.cat(grad_list, dim=0)
    std = float(grad_tensor.std(0).square().sum())
    print(f'{model.name:20}  {str(grad_tensor.shape[-1]):10}    {std:f}')
