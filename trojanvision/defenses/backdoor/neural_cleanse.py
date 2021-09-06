#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanvision.environ import env
from trojanzoo.utils import jaccard_idx, normalize_mad
from trojanzoo.utils import AverageMeter
from trojanzoo.utils import to_tensor, to_numpy, tanh_func
from trojanzoo.utils.output import prints, ansi, output_iter

import torch
import torch.optim as optim
import numpy as np

import os
import time
import datetime
import argparse
from tqdm import tqdm


class NeuralCleanse(BackdoorDefense):
    name: str = 'neural_cleanse'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--nc_epoch', type=int, help='neural cleanse optimizing epoch, defaults to 10.')
        group.add_argument('--penalize', action='store_true',
                           help='add the regularization terms, nc to tabor, defaults to False.')
        group.add_argument('--hyperparams', type=list,
                           help='the hyperparameters of  all regularization terms, defaults to [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2].')
        return group

    def __init__(self, epoch: int = 10,
                 init_cost: float = 1e-3, cost_multiplier: float = 1.5, patience: float = 10,
                 attack_succ_threshold: float = 0.99, early_stop_threshold: float = 0.99,
                 **kwargs):
        super().__init__(**kwargs)

        self.epoch: int = epoch

        self.init_cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5

        self.patience: float = patience
        self.attack_succ_threshold: float = attack_succ_threshold

        self.early_stop = True
        self.early_stop_threshold: float = early_stop_threshold
        self.early_stop_patience: float = self.patience * 2

        self.random_pos = self.attack.mark.random_pos

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.attack.mark.random_pos = False
        self.attack.mark.height_offset = 0
        self.attack.mark.width_offest = 0
        if not self.random_pos:
            self.real_mask = self.attack.mark.mask
        mark_list, mask_list, loss_list = self.get_potential_triggers()
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1)
        print('mask norms: ', mask_norms)
        print('mask MAD: ', normalize_mad(mask_norms))
        print('loss: ', loss_list)
        print('loss MAD: ', normalize_mad(loss_list))

        if not self.random_pos:
            overlap = jaccard_idx(mask_list[self.attack.target_class], self.real_mask,
                                  select_num=self.attack.mark.mark_height * self.attack.mark.mark_width)
            print(f'Jaccard index: {overlap:.3f}')

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        mark_list = [to_numpy(i) for i in mark_list]
        mask_list = [to_numpy(i) for i in mask_list]
        loss_list = [to_numpy(i) for i in loss_list]

    def get_potential_triggers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mark_list, mask_list, loss_list = [], [], []
        # todo: parallel to avoid for loop
        file_path = os.path.normpath(os.path.join(
            self.folder_path, self.get_filename(target_class=self.target_class) + '.npz'))
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            mark, mask, loss = self.remask(label)
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)
            if not self.random_pos:
                overlap = jaccard_idx(mask, self.real_mask,
                                      select_num=self.attack.mark.mark_height * self.attack.mark.mark_width)
                print(f'Jaccard index: {overlap:.3f}')
            np.savez(file_path, mark_list=[to_numpy(mark) for mark in mark_list],
                     mask_list=[to_numpy(mask) for mask in mask_list],
                     loss_list=loss_list)
            print('Defense results saved at: ' + file_path)
        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)
        return mark_list, mask_list, loss_list

    def loss_fn(self, _input, _label, Y, mask, mark, label):
        X = _input + mask * (mark - _input)
        Y = label * torch.ones_like(_label, dtype=torch.long)
        _output = self.model(X)
        return self.model.criterion(_output, Y)

    def remask(self, label: int):
        epoch = self.epoch
        # no bound
        atanh_mark = torch.randn(self.dataset.data_shape, device=env['device'])
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.dataset.data_shape[1:], device=env['device'])
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        optimizer.zero_grad()

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        for _epoch in range(epoch):
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            loader = self.dataset.loader['train']
            if env['tqdm']:
                loader = tqdm(loader)
            for data in loader:
                _input, _label = self.model.get_data(data)
                batch_size = _label.size(0)
                X = _input + mask * (mark - _input)
                Y = label * torch.ones_like(_label, dtype=torch.long)
                _output = self.model(X)

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = self.loss_fn(_input, _label, Y, mask, mark, label)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)    # (h, w)
                mark = tanh_func(atanh_mark)    # (c, h, w)
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                output_iter(_epoch + 1, epoch), **ansi).ljust(64 if env['color'] else 35)
            _str = ' '.join([
                f'Loss: {losses.avg:.4f},'.ljust(20),
                f'Acc: {acc.avg:.2f}, '.ljust(20),
                f'Norm: {norm.avg:.4f},'.ljust(20),
                f'Entropy: {entropy.avg:.4f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi) if env['tqdm'] else '', indent=4)

            # check to save best mask or not
            if acc.avg >= self.attack_succ_threshold and norm.avg < norm_best:
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                prints('up cost from %.4f to %.4f' %
                       (cost, cost * self.cost_multiplier_up), indent=4)
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                prints('down cost from %.4f to %.4f' %
                       (cost, cost / self.cost_multiplier_down), indent=4)
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = tanh_func(atanh_mask).detach()
                mark_best = tanh_func(atanh_mark).detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        self.attack.mark.mark = mark_best
        self.attack.mark.alpha_mark = mask_best
        self.attack.mark.mask = torch.ones_like(mark_best, dtype=torch.bool)
        self.attack.validate_fn()
        return mark_best, mask_best, entropy_best

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.folder_path, self.get_filename() + '.npz')
        _dict = np.load(path)
        self.attack.mark.mark = to_tensor(_dict['mark_list'][self.target_class])
        self.attack.mark.alpha_mask = to_tensor(_dict['mask_list'][self.target_class])
        self.attack.mark.mask = torch.ones_like(self.attack.mark.mark, dtype=torch.bool)
        self.attack.mark.random_pos = False
        self.attack.mark.height_offset = 0
        self.attack.mark.width_offset = 0
        print('defense results loaded from: ', path)
