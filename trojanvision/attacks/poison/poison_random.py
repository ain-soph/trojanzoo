#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import ImageModel
from trojanzoo.attacks import Attack
from trojanzoo.utils.data import TensorListDataset, dataset_to_tensor


import torch
import math
import random
import os
import argparse
from typing import Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class PoisonRandom(Attack):
    name: str = 'poison_random'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--poison_percent', type=float,
                           help='malicious training data injection probability for each batch, defaults to 0.01')
        group.add_argument('--train_mode', choices=['batch', 'dataset'],
                           help='target class of backdoor, defaults to "batch"')
        return group

    def __init__(self, poison_percent: float = 0.01, train_mode: str = 'batch', **kwargs):
        super().__init__(**kwargs)
        self.dataset: ImageSet
        self.model: ImageModel
        self.param_list['badnet'] = ['poison_percent']
        self.poison_percent: float = poison_percent
        self.poison_num = self.dataset.batch_size * self.poison_percent
        self.train_mode: str = train_mode

    def attack(self, epochs: int, save=False, **kwargs):
        if self.train_mode == 'batch':
            self.model._train(epochs, save=save,
                              validate_fn=self.validate_fn, get_data_fn=self.get_data,
                              save_fn=self.save, **kwargs)
        elif self.train_mode == 'dataset':
            dataset = self.mix_dataset()
            loader = self.dataset.get_dataloader('train', dataset=dataset)
            self.model._train(epochs, save=save,
                              validate_fn=self.validate_fn, loader_train=loader,
                              save_fn=self.save, **kwargs)

    def mix_dataset(self) -> torch.utils.data.Dataset:
        clean_set = self.dataset.loader['train'].dataset
        subset, other_set = ImageSet.split_dataset(clean_set, percent=self.poison_percent)
        if not len(subset):
            return clean_set
        _input, _label = dataset_to_tensor(subset)

        _label += torch.randint_like(_label, low=1, high=self.model.num_classes)
        _label %= self.model.num_classes
        poison_set = TensorListDataset(_input, _label.tolist())
        return torch.utils.data.ConcatDataset([poison_set, other_set])

    # ---------------------- I/O ----------------------------- #

    def get_filename(self, **kwargs):
        return f'{self.train_mode}_{self.poison_percent}'

    def save(self, filename: str = None, **kwargs):
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.model.load(file_path + '.pth')
        print('attack results loaded from: ', file_path)

    # ---------------------- Utils ---------------------------- #

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if integer:
            _label[:integer] += torch.randint_like(_label[:integer], low=1, high=self.model.num_classes)
            _label[:integer] %= self.model.num_classes
        return _input, _label

    def validate_fn(self, get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    indent: int = 0, **kwargs) -> tuple[float, float]:
        return self.model._validate(indent=indent, **kwargs)
