#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import summary

import torch
import numpy as np
import argparse
import alpsplot

eps = 1e-3
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

    data = next(iter(dataset.get_dataloader('valid', shuffle=True, batch_size=20, drop_last=True)))
    _input, _label = model.get_data(data)
    x_direct: torch.Tensor = torch.ones_like(_input)
    # x_direct /= x_direct.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)

    temp = torch.ones_like(_input).flatten(1)
    temp[:, :temp.shape[1] // 2] = -1
    temp_idx = []
    for i in range(temp.shape[0]):
        temp_idx.extend([j + i * temp.shape[0] for j in torch.randperm(temp.shape[1]).tolist()])
    x_direct = temp.flatten()[temp_idx].view_as(_input)

    temp = torch.ones_like(_input).flatten(1)
    temp[:, :temp.shape[1] // 2] = -1
    temp_idx = []
    for i in range(temp.shape[0]):
        temp_idx.extend([j + i * temp.shape[0] for j in torch.randperm(temp.shape[1]).tolist()])
    y_direct = x_direct * temp.flatten()[temp_idx].view_as(_input)

    # y_direct: torch.Tensor = torch.randn_like(x_direct)
    # y_direct = y_direct - (y_direct.flatten(1) * x_direct.flatten(1)).sum(1).view(-1, 1, 1, 1) * x_direct

    assert((y_direct.flatten(1) * x_direct.flatten(1)).sum() < 0.1)
    x_direct *= eps
    y_direct *= eps

    for num in range(len(_input)):
        print(num)

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            z = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    final_input: torch.Tensor = _input[num] + x_direct[num] * x[i][j] + y_direct[num] * y[i][j]
                    final_input = final_input.clamp(min=0.0, max=1.0)
                    z[i][j] = float(model.loss(final_input.unsqueeze(0), _label[num].view(-1)))
            return z
        fig = alpsplot.Figure(name=f'{dataset.name}_{model.name}_{num}', folder_path='./result', figsize=(5, 2.5))
        fig.set_axis_lim('x', lim=[-20 * eps, 20 * eps], piece=4, margin=[0, 0], _format='%.2f')
        fig.set_axis_lim('y', lim=[-20 * eps, 20 * eps], piece=4, margin=[0, 0], _format='%.2f')
        CS = fig.ax.contour(X * eps, Y * eps, f(X, Y))
        # fig.ax.clabel(CS, fontsize=10)
        fig.save(ext='.pdf')
        fig.save(ext='.jpg')
