#!/usr/bin/env python3

import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
from trojanvision.utils import to_numpy
import argparse
import torch, time
import numpy as np

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    model.activate_params(model.parameters())
    model.zero_grad()

    torch.random.manual_seed(int(time.time()))
    grad_list = []
    for i, data in enumerate(dataset.get_dataloader('valid', shuffle=True, batch_size=1, drop_last=True)):
        if i >= 1024:
            break
        _input, _label = model.get_data(data)
        loss = model.loss(_input, _label)
        loss.backward()
        grad_temp_list = []
        for param in model.parameters():
            grad_temp_list.append(param.grad.flatten())
        grad = torch.cat(grad_temp_list)
        grad = grad if grad.norm(p=2) <= 5.0 else grad / grad.norm(p=2) * 5.0
        grad_list.append(grad.detach().cpu().clone())
        model.zero_grad()
    
    model.eval()
    model.activate_params([])

    grad_tensor = torch.stack(grad_list)
    var = float(grad_tensor.std(dim=0).sum())
    
    print('[\'' +  model.name + '\',', str(var)+'],')
    
