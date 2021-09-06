#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import ImageModel
from trojanvision.marks import Watermark
from trojanzoo.attacks import Attack
from trojanzoo.utils import to_list
from trojanzoo.utils.data import TensorListDataset, dataset_to_list
from trojanzoo.utils.output import prints
from trojanzoo.utils.logger import SmoothedValue


import torch
import math
import random
import os
import argparse
from typing import Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class BadNet(Attack):
    r"""
    BadNet Backdoor Attack is described in detail in the paper `BadNet`_ by Tianyu Gu. 

    It attaches a fixed watermark to benign images and inject them into training set with target label.
    After retraining, the model will classify all images with watermark attached into target class.

    The authors have posted `original source code`_.

    Args:
        mark (Watermark): the attached watermark image.
        target_class (int): the target class. Default: ``0``.
        poison_percent (int): The proportion of malicious images in the training set (Max 0.5). Default: 0.1.

    .. _BadNet:
        https://arxiv.org/abs/1708.06733

    .. _original source code:
        https://github.com/Kooscii/BadNets
    """

    name: str = 'badnet'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--target_class', type=int, help='target class of backdoor, defaults to 0')
        group.add_argument('--poison_percent', type=float,
                           help='malicious training data injection probability for each batch, defaults to 0.01')
        group.add_argument('--train_mode', help='target class of backdoor, defaults to "batch"')
        return group

    def __init__(self, mark: Watermark = None, target_class: int = 0, poison_percent: float = 0.01, train_mode: str = 'batch', **kwargs):
        super().__init__(**kwargs)
        self.dataset: ImageSet
        self.model: ImageModel
        self.param_list['badnet'] = ['train_mode', 'target_class', 'poison_percent', 'poison_num']
        self.mark: Watermark = mark
        self.target_class: int = target_class
        self.poison_percent: float = poison_percent
        self.poison_num = self.dataset.batch_size * self.poison_percent
        self.train_mode: str = train_mode

    def attack(self, epoch: int, save=False, **kwargs):
        if self.train_mode == 'batch':
            self.model._train(epoch, save=save,
                              validate_fn=self.validate_fn, get_data_fn=self.get_data,
                              save_fn=self.save, **kwargs)
        elif self.train_mode == 'dataset':
            dataset = self.mix_dataset()
            loader = self.dataset.get_dataloader('train', dataset=dataset)
            self.model._train(epoch, save=save,
                              validate_fn=self.validate_fn, loader_train=loader,
                              save_fn=self.save, **kwargs)
        elif self.train_mode == 'loss':
            self.model._train(epoch, save=save,
                              validate_fn=self.validate_fn, loss_fn=self.loss_fn,
                              save_fn=self.save, **kwargs)

    def mix_dataset(self, poison_label: bool = True) -> torch.utils.data.Dataset:
        clean_dataset = self.dataset.loader['train'].dataset
        subset, _ = ImageSet.split_dataset(clean_dataset, percent=self.poison_percent)
        _input, _label = dataset_to_list(subset)
        _input = torch.stack(_input)

        if poison_label:
            _label = [self.target_class] * len(_label)
        poison_input = self.add_mark(_input)
        poison_dataset = TensorListDataset(poison_input, _label)
        return torch.utils.data.ConcatDataset([clean_dataset, poison_dataset])

    def get_filename(self, mark_alpha: float = None, target_class: int = None, **kwargs):
        if mark_alpha is None:
            mark_alpha = self.mark.mark_alpha
        if target_class is None:
            target_class = self.target_class
        mark_filename = os.path.split(self.mark.mark_path)[-1]
        mark_name, mark_ext = os.path.splitext(mark_filename)
        _file = '{mark}_tar{target:d}_alpha{mark_alpha:.2f}_mark({mark_height:d},{mark_width:d})'.format(
            mark=mark_name, target=target_class, mark_alpha=mark_alpha,
            mark_height=self.mark.mark_height, mark_width=self.mark.mark_width)
        if self.mark.random_pos:
            _file = 'random_pos_' + _file
        if self.mark.mark_distributed:
            _file = 'distributed_' + _file
        return _file

    # ---------------------- I/O ----------------------------- #

    def save(self, filename: str = None, **kwargs):
        filename = filename if filename is not None else self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.mark.save_npz(file_path + '.npz')
        self.mark.save_img(file_path + '.png')
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        filename = filename if filename is not None else self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.mark.load_npz(file_path + '.npz')
        self.model.load(file_path + '.pth')
        print('attack results loaded from: ', file_path)

    # ---------------------- Utils ---------------------------- #

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mark.add_mark(x, **kwargs)

    def loss_fn(self, _input: torch.Tensor = None, _label: torch.Tensor = None, _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        loss_clean = self.model.loss(_input, _label, **kwargs)
        poison_input = self.mark.add_mark(_input)
        poison_label = self.target_class * torch.ones_like(_label)
        loss_poison = self.model.loss(poison_input, poison_label, **kwargs)
        return (1 - self.poison_percent) * loss_clean + self.poison_percent * loss_poison

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True,
                 poison_label=True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if not keep_org:
            integer = len(_label)
        if not keep_org or integer:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input[:integer])
            _label = _label[:integer]
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label[:integer])
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, **kwargs) -> tuple[float, float]:
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        _, target_acc = self.model._validate(print_prefix='Validate Trigger Tgt', main_tag='valid trigger target',
                                             get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                             indent=indent, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org', main_tag='',
                             get_data_fn=self.get_data, keep_org=False, poison_label=False,
                             indent=indent, **kwargs)
        prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        prints(f'Neuron Jaccard Idx: {self.check_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:  # TODO: better not hardcoded
            target_acc = 0.0
        return clean_acc, target_acc

    def validate_confidence(self) -> float:
        confidence = SmoothedValue()
        with torch.no_grad():
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                idx1 = _label != self.target_class
                _input = _input[idx1]
                _label = _label[idx1]
                if len(_input) == 0:
                    continue
                poison_input = self.add_mark(_input)
                poison_label = self.model.get_class(poison_input)
                idx2 = poison_label == self.target_class
                poison_input = poison_input[idx2]
                if len(poison_input) == 0:
                    continue
                batch_conf = self.model.get_prob(poison_input)[:, self.target_class].mean()
                confidence.update(batch_conf, len(poison_input))
        return confidence.global_avg

    def check_neuron_jaccard(self, ratio=0.5) -> float:
        feats_list = []
        poison_feats_list = []
        with torch.no_grad():
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                poison_input = self.add_mark(_input)

                _feats = self.model.get_final_fm(_input)
                poison_feats = self.model.get_final_fm(poison_input)
                feats_list.append(_feats)
                poison_feats_list.append(poison_feats)
        feats_list = torch.cat(feats_list).mean(dim=0)
        poison_feats_list = torch.cat(poison_feats_list).mean(dim=0)
        length = int(len(feats_list) * ratio)
        _idx = set(to_list(feats_list.argsort(descending=True))[:length])
        poison_idx = set(to_list(poison_feats_list.argsort(descending=True))[:length])
        jaccard_idx = len(_idx & poison_idx) / len(_idx | poison_idx)
        return jaccard_idx
