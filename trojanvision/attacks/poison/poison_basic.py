#!/usr/bin/env python3

from trojanzoo.attacks import Attack
from trojanzoo.utils.output import prints

import torch
import numpy as np
import math
import random
import os
import argparse
from typing import Callable


class PoisonBasic(Attack):

    name: str = 'poison_basic'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--target_num', type=int,
                           help='number of malicious target images, defaults to 5')
        group.add_argument('--poison_percent', type=float,
                           help='malicious training data injection probability for each batch, defaults to 0.1')
        group.add_argument('--target_idx', type=int,
                           help='target label order in original classification, defaults to 1 '
                           '(0 for untargeted attack, 1 for most possible class, -1 for most unpossible class)')
        group.add_argument('--clean_epoch', type=int,
                           help='number of training epochs using clean data after attack, defaults to 0')
        return group

    def __init__(self, target_num: int = 5, poison_percent: float = 0.1, target_idx: int = 1,
                 clean_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.param_list['poison'] = ['target_num', 'poison_percent', 'poison_num', 'target_idx']
        self.target_num = target_num
        self.poison_percent = poison_percent
        self.target_idx = target_idx
        self.clean_epoch = clean_epoch

        self.temp_input: torch.Tensor = None
        self.temp_label: torch.Tensor = None

        self.poison_num = self.dataset.batch_size * self.poison_percent

    def attack(self, epochs: int, **kwargs):
        # model._validate()
        total = 0
        target_conf_list = []
        target_acc_list = []
        clean_acc_list = []

        validset = self.dataset.get_dataset('valid')
        testset, _ = self.dataset.split_dataset(validset, percent=0.3)
        loader = self.dataset.get_dataloader(mode='valid', dataset=testset,
                                             batch_size=2 * self.target_num,
                                             shuffle=True, drop_last=True)
        for data in loader:
            if total >= 10:
                break
            self.model.load()
            _input, _label = self.model.remove_misclassify(data)
            if len(_input) < self.target_num:
                continue
            _input = _input[:self.target_num]
            _label = self.model.generate_target(_input, idx=self.target_idx)
            self._train(_input=_input, _label=_label, epochs=epochs, **kwargs)
            if self.clean_epoch > 0:
                self.model._train(epochs=self.clean_epoch, indent=4, validate_fn=self.validate_fn, **kwargs)
            target_conf, target_acc = self.validate_target()
            clean_acc, _ = self.model._validate()
            target_conf_list.append(target_conf)
            target_acc_list.append(target_acc)
            clean_acc_list.append(clean_acc)
            total += 1
            print(f'[{total}/10]\n'
                  f'target confidence: {np.mean(target_conf_list)}({np.std(target_conf_list)})\n'
                  f'target accuracy: {np.mean(target_acc_list)}({np.std(target_acc_list)})\n'
                  f'clean accuracy: {np.mean(clean_acc_list)}({np.std(clean_acc_list)})\n\n\n')

    def _train(self, _input: torch.Tensor, _label: torch.Tensor, epochs: int, save=False, indent=0, **kwargs):
        self.temp_input = _input
        self.temp_label = _label

        return self.model._train(epochs=epochs, save=save,
                                 get_data_fn=self.get_data, save_fn=self.save,
                                 validate_fn=self.validate_fn, indent=indent + 4, **kwargs)

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True, poison_label=True,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if not keep_org or integer:
            org_input, org_label = _input, _label
            quotient = integer // self.target_num
            remainder = integer % self.target_num
            idx = torch.randperm(self.target_num)[:remainder]
            try:
                input_list = [self.temp_input] * quotient + [self.temp_input[idx]]
                label_list = [self.temp_label] * quotient + [self.temp_label[idx]]
            except Exception:
                print(self.target_num, integer, self.temp_input.shape, self.temp_label.shape)
                print(idx)
                exit()
            _input = torch.cat(input_list)
            if poison_label:
                _label = torch.cat(label_list)
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    # def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
    #     clean = self.model.loss(x, y, **kwargs)
    #     training: bool = self.model._model.training
    #     self.model.eval()
    #     poison = self.model.loss(self.temp_input, self.temp_label)
    #     self.model.train(mode=training)
    #     print(f'clean: {clean:7.5f}    poison: {poison:7.5f}   eval: {poison:7.5f}')
    #     loss = (1 - self.poison_percent) * self.model.loss(x, y, **kwargs) \
    #         + self.poison_percent * self.model.loss(self.temp_input, self.temp_label)
    #     return loss

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def get_filename(self, **kwargs):
        return self.model.name

    def validate_target(self, indent: int = 0, verbose=True) -> tuple[float, float]:
        self.model.eval()
        _output = self.model(self.temp_input)
        asr, _ = self.model.accuracy(_output, self.temp_label, topk=(1, 5))
        conf = float(self.model.get_target_prob(self.temp_input, self.temp_label).mean())
        target_loss = self.model.loss(self.temp_input, self.temp_label)
        if verbose:
            prints(f'Validate Target:       Loss: {target_loss:10.4f}     '
                   f'Confidence: {conf:10.4f}    ASR: {asr:7.3f}',
                   indent=indent)
        return asr, conf

    def validate_fn(self, get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    main_tag: str = 'valid', indent: int = 0, verbose=True,
                    loss_fn: Callable = None, **kwargs) -> tuple[float, float]:
        asr, _ = self.validate_target(indent=indent, verbose=verbose)
        clean_acc, = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                          get_data_fn=None, indent=indent, **kwargs)
        return asr, clean_acc
