# -*- coding: utf-8 -*-

from package.imports.universal import *

from package.utils.utils import *

import argparse


class Neural_Cleanse():

    def __init__(self, model, name='neural_cleanse', epoch=10, init_cost=1e-3, cost_multiplier=1.5, **kwargs):
        self.model = model
        self.name = name

        self.trainloader = self.model.dataset.loader['train']
        for data in self.trainloader:
            _input, _ = self.model.get_data(data, mode='train')
            break
        self.shape = list(_input.shape)
        self.shape[0] = 1

        self.epoch = epoch
        self.early_stop = True
        self.attack_succ_threshold = 0.99
        self.early_stop_threshold = 0.99
        self.patience = 10
        self.early_stop_patience = self.patience*2
        self.verbose = 1
        self.epsilon = 1e-7

        self.init_cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5

    def get_potential_triggers(self):
        trigger_list = []
        mask_list = []
        loss_ce_list = []
        for label in range(self.model.num_classes):
            # print('label: ', label)
            print(label)
            print(self.model.num_classes)
            triggers, mask, loss_ce = self.get_potential_triggers_for_label(
                label)
            trigger_list.append(triggers)
            mask_list.append(mask)
            loss_ce_list.append(loss_ce)
        trigger_list = to_tensor(trigger_list)
        mask_list = to_tensor(mask_list)
        loss_ce_list = to_tensor(loss_ce_list)

        return trigger_list, mask_list, loss_ce_list

    def get_potential_triggers_for_label(self, label):
        # no bound
        atanh_triggers = to_tensor(torch.zeros(self.shape))
        atanh_triggers.requires_grad = True
        atanh_mask = to_tensor(torch.zeros(self.shape))
        atanh_mask.requires_grad = True

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        reg_best = float('inf')
        mask_best = None
        pattern_best = None
        loss_ce_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        optimizer = optim.Adam(
            [atanh_triggers, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        criterion = nn.CrossEntropyLoss(reduction='none')
        for step in range(self.epoch):
            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for j, (X, Y) in enumerate(self.trainloader):
                batch_num = len(Y)
                mask = torch.tanh(atanh_mask)    # (1, c, h, w)
                triggers = torch.tanh(atanh_triggers)    # (1, c, h, w)
                X = to_tensor(X)    # (batch_num, c, h, w)
                X.requires_grad = True

                _input = ((1-mask)*X+mask*triggers)
                _label = to_tensor(np.repeat(label, batch_num))
                _output = self.model(_input)

                _result = _output.max(1)[1]
                loss_acc = _label.eq(_result).float()

                loss_ce = criterion(_output, _label)
                loss_reg = float(mask.norm(p=1))
                loss = loss_ce+cost*loss_reg

                loss_mean = loss.mean()
                loss_mean.backward()

                loss_ce_list.extend(loss_ce.detach().tolist())
                loss_reg_list.append(loss_reg)
                loss_list.extend(loss.detach().tolist())
                loss_acc_list.extend(loss_acc.detach().tolist())

                optimizer.step()
                optimizer.zero_grad()

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = torch.tanh(atanh_mask).add(1).div(2)
                pattern_best = torch.tanh(atanh_triggers).add(1).div(2)
                reg_best = avg_loss_reg
                loss_ce_best = avg_loss_ce
            # verbose
            # if self.verbose != 0:
            #     if self.verbose == 2 or step % (self.epoch // 10) == 0:
            #         print('step: %3d, cost: %.5f, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
            #               (step, cost, avg_loss_acc, avg_loss,
            #                avg_loss_ce, avg_loss_reg, reg_best))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
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
                if self.verbose == 2:
                    print('up cost from %.2f to %.2f' %
                          (cost, cost * self.cost_multiplier_up))
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2f to %.2f' %
                          (cost, cost / self.cost_multiplier_down))
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = torch.tanh(atanh_mask).add(1).div(2)
                pattern_best = torch.tanh(atanh_triggers).add(1).div(2)
                reg_best = avg_loss_reg
                loss_ce_best = avg_loss_ce

        return pattern_best, mask_best, loss_ce_best

    def get_mask_norms(self, mask):
        return mask.view(self.model.num_classes, -1).norm(p=1, dim=1)

    def normalize_mask_norms(self, mask_norms):
        median = mask_norms.median()
        abs_dev = (mask_norms-median).abs()
        mad = abs_dev.mean()

        measures = abs_dev/mad/1.4826

        return measures

    def measure_triggers(self, mask):
        mask_norms = self.get_mask_norms(mask)
        measures = self.normalize_mask_norms(mask_norms)
        return measures
