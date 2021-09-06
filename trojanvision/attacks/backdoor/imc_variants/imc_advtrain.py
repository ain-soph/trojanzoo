#!/usr/bin/env python3

from trojanvision.attacks.adv import PGD
from trojanvision.attacks.backdoor.imc import IMC
from trojanvision.environ import env
from trojanzoo.utils.output import output_iter, ansi, prints
from trojanzoo.utils import AverageMeter

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import time
import datetime
import os
from tqdm import tqdm
from collections.abc import Callable


class IMC_AdvTrain(IMC):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang.

    Based on :class:`trojanzoo.attacks.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc_advtrain'

    def __init__(self, pgd_alpha: float = 2.0 / 255, pgd_eps: float = 8.0 / 255, pgd_iter: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.param_list['adv_train'] = ['pgd_alpha', 'pgd_eps', 'pgd_iter']
        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.pgd_iter = pgd_iter
        self.pgd = PGD(pgd_alpha=pgd_alpha, pgd_eps=pgd_eps, iteration=pgd_iter,
                       target_idx=0, stop_threshold=None, model=self.model, dataset=self.dataset)

    def get_poison_data(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if integer:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input[:integer])
            _label = self.target_class * torch.ones_like(org_label[:integer])
        return _input, _label

    def attack(self, epoch: int, save=False, **kwargs):
        self.adv_train(epoch, save=save,
                       validate_fn=self.validate_fn, get_data_fn=self.get_data,
                       epoch_fn=self.epoch_fn, save_fn=self.save, **kwargs)

    def adv_train(self, epoch: int, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler._LRScheduler = None,
                  validate_interval=10, save=False, verbose=True, indent=0, epoch_fn: Callable = None,
                  **kwargs):
        loader_train = self.dataset.loader['train']
        file_path = os.path.join(self.folder_path, self.get_filename() + '.pth')

        _, best_acc = self.validate_fn(verbose=verbose, indent=indent, **kwargs)

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        params: list[nn.Parameter] = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        for _epoch in range(epoch):
            if callable(epoch_fn):
                self.model.activate_params([])
                epoch_fn()
                self.model.activate_params(params)
            losses.reset()
            top1.reset()
            top5.reset()
            epoch_start = time.perf_counter()
            if verbose and env['tqdm']:
                loader_train = tqdm(loader_train)
            optimizer.zero_grad()
            for data in loader_train:
                _input, _label = self.model.get_data(data)
                noise = torch.zeros_like(_input)

                poison_input, poison_label = self.get_poison_data(data)

                adv_x = _input
                self.model.train()
                loss = self.model.loss(adv_x, _label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                for m in range(self.pgd.iteration):
                    self.model.eval()
                    adv_x, _ = self.pgd.optimize(_input=_input, noise=noise,
                                                 target=_label, iteration=1)

                    optimizer.zero_grad()
                    self.model.train()

                    x = torch.cat((adv_x, poison_input))
                    y = torch.cat((_label, poison_label))
                    loss = self.model.loss(x, y)
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    _output = self.model.get_logits(_input)
                acc1, acc5 = self.model.accuracy(_output, _label, topk=(1, 5))
                batch_size = int(_label.size(0))
                losses.update(loss.item(), batch_size)
                top1.update(acc1, batch_size)
                top5.update(acc5, batch_size)
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            self.model.eval()
            self.model.activate_params([])
            if verbose:
                pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                    output_iter(_epoch + 1, epoch), **ansi).ljust(64 if env['color'] else 35)
                _str = ' '.join([
                    f'Loss: {losses.avg:.4f},'.ljust(20),
                    f'Top1 Clean Acc: {top1.avg:.3f}, '.ljust(30),
                    f'Top5 Clean Acc: {top5.avg:.3f},'.ljust(30),
                    f'Time: {epoch_time},'.ljust(20),
                ])
                prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi) if env['tqdm'] else '',
                       indent=indent)
            if lr_scheduler:
                lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch + 1) % validate_interval == 0 or _epoch == epoch - 1:
                    _, cur_acc = self.validate_fn(verbose=verbose, indent=indent, **kwargs)
                    if cur_acc < best_acc:
                        prints('{purple}best result update!{reset}'.format(**ansi), indent=indent)
                        prints(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}', indent=indent)
                        best_acc = cur_acc
                    if save:
                        self.save()
                    if verbose:
                        print('-' * 50)
        self.model.zero_grad()
