# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor
from trojanzoo.optim.pgd import PGD
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.utils.model import AverageMeter

import torch
from torch import optim

import time
import datetime
from tqdm import tqdm

from trojanzoo.utils.config import Config
env = Config.env


class Adv_Train(Defense_Backdoor):

    name: str = 'adv_train'

    def __init__(self, pgd_alpha: float = 2.0 / 255, pgd_epsilon: float = 8.0 / 255, pgd_iteration: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.param_list['adv_train'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
        self.pgd_alpha = pgd_alpha
        self.pgd_epsilon = pgd_epsilon
        self.pgd_iteration = pgd_iteration
        self.pgd = PGD(alpha=pgd_alpha, epsilon=pgd_epsilon, iteration=pgd_iteration, stop_threshold=None)
        _, self.clean_acc, _ = self.model._validate(print_prefix='Baseline Clean', get_data=None, **kwargs)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        print()
        self.adv_train(verbose=True, **kwargs)
        self.attack.validate_func()

    def validate_func(self, get_data=None, **kwargs) -> (float, float, float):
        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        get_data=None, **kwargs)
        adv_loss, adv_acc, _ = self.model._validate(print_prefix='Validate Adv',
                                                    get_data=self.get_data, **kwargs)
        # todo: Return value
        if self.clean_acc - clean_acc > 20 and self.clean_acc > 40:
            adv_acc = 0.0
        return clean_loss + adv_loss, adv_acc, clean_acc

    def get_data(self, data: (torch.Tensor, torch.LongTensor), **kwargs) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)

        def loss_fn(X: torch.FloatTensor):
            return -self.model.loss(X, _label)
        adv_x, _ = self.pgd.optimize(_input=_input, loss_fn=loss_fn)
        return adv_x, _label

    def save(self, **kwargs):
        self.model.save(folder_path=self.folder_path, suffix='_adv_train', verbose=True, **kwargs)

    def adv_train(self, epoch: int, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler._LRScheduler = None,
                  validate_interval=10, save=False, suffix: str = None, verbose=True, indent=0,
                  **kwargs):
        loader_train = self.dataset.loader['train']

        _, best_acc, _ = self.validate_func(verbose=verbose, indent=indent, **kwargs)

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        params = [param_group['params'] for param_group in optimizer.param_groups]
        self.activate_params(params)
        optimizer.zero_grad()
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            epoch_start = time.perf_counter()
            if verbose and env['tqdm']:
                loader_train = tqdm(loader_train)
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
                    if cur_acc >= best_acc:
                        prints('best result update!', indent=indent)
                        prints(f'Current Acc: {cur_acc:.3f}    Best Acc: {best_acc:.3f}', indent=indent)
                        best_acc = cur_acc
                        if save:
                            self.save(suffix=suffix, verbose=verbose)
                    if verbose:
                        print('-' * 50)
        self.model.zero_grad()
        self.model.eval()
        self.model.activate_params([])
