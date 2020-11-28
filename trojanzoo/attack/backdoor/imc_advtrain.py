# -*- coding: utf-8 -*-

from .imc import IMC

import torch

import math
import random
from typing import Dict, Tuple
from trojanzoo.utils.model import AverageMeter

import time
import datetime
from tqdm import tqdm
from typing import Tuple

from trojanzoo.utils.config import Config
env = Config.env

class IMC_AdvTrain(IMC):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang.

    Based on :class:`trojanzoo.attack.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc_advtrain'

    def get_poison_data(self, data: Tuple[torch.Tensor, torch.LongTensor], **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
        _input, _label = self.model.get_data(data)
        integer = len(_label)
        if poison_label:
            _label = self.target_class * torch.ones_like(_label[:integer])
        else:
            _label = _label[:integer]
        return _input, _label

    def validate_func(self, get_data=None, loss_fn=None, **kwargs) -> Tuple[float, float, float]:
        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        get_data=None, **kwargs)
        target_loss, target_acc, _ = self.model._validate(print_prefix='Validate Trigger Tgt',
                                                          get_data=self.get_poison_data, **kwargs)
        _, orginal_acc, _ = self.model._validate(print_prefix='Validate Trigger Org',
                                                 get_data=self.get_poison_data, poison_label=False, **kwargs)
        self.model._validate(print_prefix='Validate STRIP Tgt',
                             get_data=self.get_poison_data, strip=True, **kwargs)
        self.model._validate(print_prefix='Validate STRIP Org',
                             get_data=self.get_poison_data, strip=True, poison_label=False, **kwargs)
        print(f'Validate Confidence : {self.validate_confidence():.3f}')
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc


    def adv_train(self, epoch: int, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler._LRScheduler = None,
                  validate_interval=10, save=False, verbose=True, indent=0,
                  **kwargs):
        loader_train = self.dataset.loader['train']
        file_path = self.folder_path + self.get_filename() + '.pth'

        _, best_acc, _ = self.validate_func(verbose=verbose, indent=indent, **kwargs)

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        params = [param_group['params'] for param_group in optimizer.param_groups]
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            epoch_start = time.perf_counter()
            if verbose and env['tqdm']:
                loader_train = tqdm(loader_train)
            self.model.activate_params(params)
            optimizer.zero_grad()
            for data in loader_train:
                _input, _label = self.model.get_data(data)
                noise = torch.zeros_like(_input)

                def loss_fn(X: torch.FloatTensor):
                    return -self.model.loss(X, _label)
                adv_x = _input
                self.model.train()
                loss = self.model.loss(adv_x, _label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                for m in range(self.pgd.iteration):
                    self.model.eval()
                    adv_x, _ = self.pgd.optimize(_input=_input, noise=noise, loss_fn=loss_fn, iteration=1)

                    optimizer.zero_grad()
                    self.model.train()
                    loss = self.model.loss(adv_x, _label)
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
                    _, cur_acc, _ = self.validate_func(verbose=verbose, indent=indent, **kwargs)
                    if cur_acc < best_acc:
                        prints('best result update!', indent=indent)
                        prints(f'Current Acc: {cur_acc:.3f}    Best Acc: {best_acc:.3f}', indent=indent)
                        best_acc = cur_acc
                    if save:
                        self.model.save(file_path=file_path, verbose=verbose)
                    if verbose:
                        print('-' * 50)
        self.model.zero_grad()
