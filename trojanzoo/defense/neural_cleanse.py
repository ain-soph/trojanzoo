# -*- coding: utf-8 -*-

# todo: search_epoch rename

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.process import Process

from trojanzoo.utils import to_list
from trojanzoo.utils.output import output_iter
from trojanzoo.optim.uname import Uname


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from typing import List

from trojanzoo.utils.config import Config
env = Config.env


class Neural_Cleanse():

    name = 'neural_cleanse'

    def __init__(self, dataset: ImageSet, model: ImageModel, data_shape: List[int], search_epoch: int = 50,
                 init_cost: float = 1e-3, cost_multiplier: float = 1.5, patience: float = 10,
                 attack_succ_threshold: float = 0.99, early_stop_threshold: float = 0.99, **kwargs):
        self.data_shape: List[int] = data_shape
        self.dataset: ImageSet = dataset
        self.model: ImageModel = model

        self.search_epoch: int = search_epoch

        self.init_cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5

        self.patience: float = patience
        self.attack_succ_threshold: float = attack_succ_threshold

        self.early_stop = True
        self.early_stop_threshold: float = early_stop_threshold
        self.early_stop_patience: float = self.patience * 2

    def get_potential_triggers(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mark_list, mask_list, loss_ce_list = [], [], []
        for label in range(self.model.num_classes):
            # print('label: ', label)
            print('Class: ', output_iter(label, self.model.num_classes))
            mark, mask, loss_ce = self.get_potential_triggers_for_label(
                label)
            mark_list.append(mark)
            mask_list.append(mask)
            loss_ce_list.append(loss_ce)
        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_ce_list = torch.as_tensor(loss_ce_list)

        return mark_list, mask_list, loss_ce_list

    def get_potential_triggers_for_label(self, label: int):
        # no bound
        atanh_mark = torch.randn(self.data_shape, device=env['device'])
        atanh_mark.requires_grad = True
        atanh_mask = torch.randn(self.data_shape, device=env['device'])
        atanh_mask.requires_grad = True
        mask = Uname.tanh_func(atanh_mask)    # (1, c, h, w)
        mark = Uname.tanh_func(atanh_mark)    # (1, c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer.zero_grad()

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        reg_best = float('inf')
        mask_best = None
        mark_best = None
        loss_ce_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        for step in range(self.search_epoch):
            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for data in tqdm(self.dataset.loader['train']):
                _input, _label = self.model.get_data(data)
                X = (1 - mask) * _input + mask * mark
                Y = label * torch.ones_like(_label, dtype=torch.long)

                _output = self.model(X)
                _result = _output.max(1)[1]

                loss_acc = Y.eq(_result).float()
                loss_ce = criterion(_output, Y)
                loss_reg = float(mask.norm(p=1))
                loss = loss_ce + cost * loss_reg
                loss_mean = loss.mean()

                loss_ce_list.extend(to_list(loss_ce))
                loss_reg_list.append(loss_reg)
                loss_list.extend(to_list(loss))
                loss_acc_list.extend(to_list(loss_acc))

                loss_mean.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = Uname.tanh_func(atanh_mask)    # (1, c, h, w)
                mark = Uname.tanh_func(atanh_mark)    # (1, c, h, w)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = mask.detach()
                mark_best = mark.detach()
                reg_best = avg_loss_reg.detach()
                loss_ce_best = avg_loss_ce.detach()

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
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

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                print('up cost from %.2f to %.2f' %
                      (cost, cost * self.cost_multiplier_up))
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                print('down cost from %.2f to %.2f' %
                      (cost, cost / self.cost_multiplier_down))
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = Uname.tanh_func(atanh_mask).detach()
                mark_best = Uname.tanh_func(atanh_mark).detach()
                reg_best = avg_loss_reg.detach()
                loss_ce_best = avg_loss_ce.detach()

        return mark_best, mask_best, loss_ce_best
