#!/usr/bin/env python3

import trojanvision

import torch
import argparse

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model)
    # loss, acc1 = model._validate()

    model.activate_params(model.parameters())
    model.zero_grad()

    torch.random.manual_seed(int(time.time()))
    grad_x = None
    grad_xx = None
    n_sample = 512

    loader = dataset.get_dataloader('valid', shuffle=True,
                                    batch_size=1, drop_last=True)
    for i, data in enumerate(loader):
        if i >= n_sample:
            break
        _input, _label = model.get_data(data)
        loss = model.loss(_input, _label)
        loss.backward()
        grad_temp_list = []
        for param in model.parameters():
            grad_temp_list.append(param.grad.flatten())
        grad = torch.cat(grad_temp_list)
        grad = grad if grad.norm(p=2) <= 5.0 else grad / grad.norm(p=2) * 5.0
        grad_temp = grad.detach().cpu().clone()
        if grad_x is None:
            grad_x = grad_temp / n_sample
            grad_xx = grad_temp.square() / n_sample
        else:
            grad_x += grad_temp / n_sample
            grad_xx += grad_temp.square() / n_sample
        model.zero_grad()

    model.eval()
    model.activate_params([])

    grad_arr = grad_xx - grad_x.square()
    grad_arr[grad_arr < 0] = 0
    var: float = grad_arr.sqrt().sum().item()

    print(f'{model.name:20}  {var:f}')
