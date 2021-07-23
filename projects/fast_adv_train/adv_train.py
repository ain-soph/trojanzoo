#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python examples/train.py --verbose 1 --color --epoch 200 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --dataset cifar10 --model resnet18_comp

import trojanvision
from trojanvision.utils import summary

import torch
import numpy as np
import argparse


class Cyclic_Scheduler():
    def __init__(self, optimizer: torch.optim.Optimizer,
                 epoch: int, lr_max: float, loader_length: int) -> None:
        self.optimizer = optimizer
        self.epoch = epoch
        self.lr_max = lr_max
        self.loader_length = loader_length
        self._step = 0

    def step(self):
        self._step += 1
        self.optimizer.param_groups[0]['lr'] = self.get_lr(self._step / self.loader_length)

    def get_lr(self, t: float) -> float:
        return np.interp([t], [0, self.epoch * 2 // 5, self.epoch], [0, self.lr_max, 0])[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    dataset.norm_par = None
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

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

    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)

    if isinstance(trainer.optimizer, torch.optim.Adam):
        trainer.lr_scheduler = Cyclic_Scheduler(trainer.optimizer, epoch=args.epoch,
                                                lr_max=args.lr, loader_length=len(dataset.loader['train']))
    else:
        lr_steps = args.epoch * len(dataset.loader['train'])
        trainer.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(trainer.optimizer,
                                                                 base_lr=0.0, max_lr=args.lr,
                                                                 step_size_up=lr_steps // 2,
                                                                 step_size_down=lr_steps // 2)
    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)
    # model._train(lr_scheduler_freq='step', **trainer)

    args.epochs = args.epoch
    args.lr_max = args.lr
    args.attack = 'fgsm'
    args.alpha = args.adv_train_alpha
    args.epsilon = args.adv_train_eps
    args.lr_type = 'cyclic'
    train_loader = dataset.loader['train']
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    import torch.nn.functional as F
    criterion = nn.CrossEntropyLoss()

    if args.lr_type == 'cyclic':
        def lr_schedule(t):
            return np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat':
        def lr_schedule(t):
            return args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    # model._validate()
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                delta = delta.detach()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            elif args.attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1 - X, delta.data), 0 - X)
                for _ in range(args.attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)[I]
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
        print(epoch)
        model._validate()
        model.train()
