#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --epochs 200 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --dataset cifar10 --model resnet18_comp

import trojanvision

import torch
import numpy as np
import argparse


class Cyclic_Scheduler():
    def __init__(self, optimizer: torch.optim.Optimizer,
                 epochs: int, lr_max: float, loader_length: int) -> None:
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_max = lr_max
        self.loader_length = loader_length
        self._step = 0

    def step(self):
        self._step += 1
        self.optimizer.param_groups[0]['lr'] = self.get_lr(self._step / self.loader_length)

    def get_lr(self, t: float) -> float:
        return np.interp([t], [0, self.epochs * 2 // 5, self.epochs], [0, self.lr_max, 0])[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    dataset.norm_par = None
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    import torch.nn as nn
    model._model.features = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU()).cuda()
    model._model.pool = nn.Identity()
    model._model.classifier = nn.Sequential(
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10)).cuda()

    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if isinstance(trainer.optimizer, torch.optim.Adam):
        trainer.lr_scheduler = Cyclic_Scheduler(trainer.optimizer, epochs=kwargs['epochs'],
                                                lr_max=kwargs['lr'], loader_length=len(dataset.loader['train']))
    else:
        lr_steps = kwargs['epochs'] * len(dataset.loader['train'])
        trainer.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(trainer.optimizer,
                                                                 base_lr=0.0, max_lr=kwargs['lr'],
                                                                 step_size_up=lr_steps // 2,
                                                                 step_size_down=lr_steps // 2)
    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
    # model._train(lr_scheduler_freq='step', **trainer)

    # kwargs['epochs = kwargs['epochs
    kwargs['lr_max'] = kwargs['lr']
    kwargs['attack'] = 'fgsm'
    kwargs['alpha'] = kwargs['adv_train_alpha']
    kwargs['epsilon'] = kwargs['adv_train_eps']
    kwargs['lr_type'] = 'cyclic'
    train_loader = dataset.loader['train']
    opt = torch.optim.Adam(model.parameters(), lr=kwargs['lr_max'])
    import torch.nn.functional as F
    criterion = nn.CrossEntropyLoss()

    if kwargs['lr_type'] == 'cyclic':
        def lr_schedule(t):
            return np.interp([t], [0, kwargs['epochs'] * 2 // 5, kwargs['epochs']], [0, kwargs['lr_max'], 0])[0]
    elif kwargs['lr_type'] == 'flat':
        def lr_schedule(t):
            return kwargs['lr_max']
    else:
        raise ValueError('Unknown lr_type')

    # model._validate()
    model.train()
    for _epoch in range(kwargs['epochs']):
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(_epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if kwargs['attack'] == 'fgsm':
                delta = torch.zeros_like(X).uniform_(-kwargs['epsilon'], kwargs['epsilon']).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + kwargs['alpha'] * torch.sign(grad), -kwargs['epsilon'], kwargs['epsilon'])
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                delta = delta.detach()
            elif kwargs['attack'] == 'none':
                delta = torch.zeros_like(X)
            elif kwargs['attack'] == 'pgd':
                delta = torch.zeros_like(X).uniform_(-kwargs['epsilon'], kwargs['epsilon'])
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                for _ in range(kwargs['attack_iters']):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + kwargs['alpha'] * torch.sign(grad), -kwargs['epsilon'], kwargs['epsilon'])[I]
                    delta.data[I] = torch.max(torch.min(1 - X, delta.data), 0 - X)[I]
                delta = delta.detach()
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        print(_epoch)
        model._validate()
        model.train()
