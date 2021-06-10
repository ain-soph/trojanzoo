#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import summary

import torch
import numpy as np
import argparse
import alpsplot

import torch.nn as nn

eps = 2.5e-2
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
    loss, acc1 = model._validate()

    # 生成x,y的数据
    n = 41
    x = np.linspace(-20, 20, n)
    y = np.linspace(-20, 20, n)

    # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
    X, Y = np.meshgrid(x, y)

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data = next(iter(dataset.get_dataloader('valid', shuffle=True, batch_size=400, drop_last=True)))
    _input, _label = model.get_data(data)

    _input = _input.view(20, 20, *(_input.shape[1:]))
    _label = _label.view(20, 20)
    # layer = model._model.classifier.fc.weight
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            layer = module.weight
            break

    param: torch.Tensor = layer.data.clone()
    # x_direct: torch.Tensor = torch.ones_like(param)
    # x_direct /= x_direct.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)

    temp = torch.ones_like(param).flatten()
    temp[:temp.shape[0] // 2] = -1
    temp_idx = [j for j in torch.randperm(temp.shape[0]).tolist()]
    x_direct = temp.flatten()[temp_idx].view_as(param)

    temp = torch.ones_like(param).flatten()
    temp[:temp.shape[0] // 2] = -1
    temp_idx = [j for j in torch.randperm(temp.shape[0]).tolist()]
    y_direct = x_direct * temp.flatten()[temp_idx].view_as(param)

    assert((y_direct.flatten() * x_direct.flatten()).sum() < 0.1)
    x_direct *= eps
    y_direct *= eps

    for num in range(len(_input)):
        print(num)

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            z = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    layer.data = param + x_direct * x[i][j] + y_direct * y[i][j]
                    z[i][j] = float(model.loss(_input[num], _label[num]))
            return z
        fig = alpsplot.Figure(name=f'{dataset.name}_{model.name}_{num}', folder_path='./result', figsize=(5, 2.5))
        fig.set_axis_lim('x', lim=[-20 * eps, 20 * eps], piece=4, margin=[0, 0], _format='%.2f')
        fig.set_axis_lim('y', lim=[-20 * eps, 20 * eps], piece=4, margin=[0, 0], _format='%.2f')
        CS = fig.ax.contour(X * eps, Y * eps, f(X, Y))
        # fig.ax.clabel(CS, fontsize=14)
        fig.save(ext='.pdf')
        fig.save(ext='.jpg')
