#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imc import IMC
from trojanvision.marks import Watermark

import torch
import torch.utils.data
import numpy as np
import math
import random


class IMC_Multi(IMC):
    name: str = 'imc_multi'

    def __init__(self, mark_num: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.mark_list: list[tuple[Watermark, int]] = []
        for i in range(mark_num):
            height_offset = random.randint(0, self.mark.data_shape[-2] - self.mark.mark_height)
            width_offset = random.randint(0, self.mark.data_shape[-1] - self.mark.mark_width)
            mark = Watermark(data_shape=self.mark.data_shape, edge_color=self.mark.edge_color, mark_path=self.mark.mark_path,
                             mark_alpha=self.mark.mark_alpha, mark_height=self.mark.mark_height, mark_width=self.mark.mark_width,
                             height_offset=height_offset, width_offset=width_offset,
                             random_init=True, random_pos=False, mark_distributed=self.mark.mark_distributed)
            self.mark_list.append((mark, i))

    def attack(self, epoch: int, save=False, **kwargs):
        self.model._train(epoch, save=save,
                          validate_func=self.validate_func_multi, get_data_fn=self.get_train_data,
                          save_fn=self.save, **kwargs)

    # ---------------------- I/O ----------------------------- #

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        np.save(file_path + '.npy', self.mark_list)
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        print('attack results loaded from: ', file_path)
        self.mark_list = np.load(file_path + '.npy', allow_pickle=True)
        self.model.load(file_path + '.pth')

    # ---------------------- Utils ---------------------------- #

    def get_train_data(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        input_list = [_input]
        label_list = [_label]
        for mark, target_class in self.mark_list:
            decimal, integer = math.modf(self.poison_num)
            integer = int(integer)
            if random.uniform(0, 1) < decimal:
                integer += 1
            if not integer:
                continue
            poison_input = mark.add_mark(_input[:integer])
            poison_label = target_class * torch.ones_like(_label[:integer])
            input_list.append(poison_input)
            label_list.append(poison_label)
        _input = torch.cat(input_list)
        _label = torch.cat(label_list)
        return _input, _label

    def get_poison_data(self, data: tuple[torch.Tensor, torch.Tensor], mark: Watermark = None, target_class: int = 0, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        poison_input = mark.add_mark(_input)
        poison_label = target_class * torch.ones_like(_label)
        return poison_input, poison_label

    def validate_func_multi(self, get_data_fn=None, loss_fn=None, **kwargs) -> tuple[float, float, float]:
        clean_loss, clean_acc = self.model._validate(print_prefix='Validate Clean',
                                                        get_data_fn=None, **kwargs)
        target_acc = 100.0
        target_loss = 0.0
        for i, (mark, target_class) in enumerate(self.mark_list):
            loss, acc = self.model._validate(print_prefix=f'Validate Trigger {i} target {target_class} ', get_data_fn=self.get_poison_data,
                                                mark=mark, target_class=target_class, **kwargs)
            target_loss = max(loss, target_loss)
            target_acc = min(acc, target_acc)
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc
