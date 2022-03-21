#!/usr/bin/env python3

from trojanvision.environ import env
from trojanzoo.defenses import Defense
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.metric import mask_jaccard, normalize_mad
from trojanzoo.utils.output import output_iter, prints
from trojanzoo.utils.tensor import tanh_func
from trojanzoo.utils.data import TensorListDataset, sample_batch

import torch
import torch.optim as optim
import numpy as np
from sklearn import metrics

import os
from abc import abstractmethod

from typing import TYPE_CHECKING
from trojanvision.datasets import ImageSet
from trojanvision.models import ImageModel
from trojanvision.attacks.backdoor import BadNet
import argparse
from collections.abc import Iterable
if TYPE_CHECKING:
    import torch.utils.data    # TODO: python 3.10


def format_list(_list: list, _format: str = ':8.3f') -> str:
    return '[' + ', '.join(['{{{}}}'.format(_format).format(a) for a in _list]) + ']'


class BackdoorDefense(Defense):
    r"""Backdoor defense abstract class.
    It inherits :class:`trojanzoo.defenses.Defense`.

    Args:
        original (bool): Whether to load original clean model.
            If ``False``, load attack poisoned model
            by calling ``self.attack.load()``.

    Attributes:
        real_mark (torch.Tensor): Watermark that the attacker uses
            with shape ``(C+1, H, W)``.
        real_mask (torch.Tensor): Mask of the watermark
            by calling :meth:`trojanvision.marks.Watermark.get_mask()`.
    """
    name: str = 'backdoor_defense'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--original', action='store_true',
                           help='whether to load original clean model '
                           '(default: False)')
        return group

    def __init__(self, attack: BadNet, original: bool = False, **kwargs):
        self.original: bool = original
        if not self.original:
            attack.load(**kwargs)
        super().__init__(attack=attack, **kwargs)
        self.dataset: ImageSet
        self.model: ImageModel
        self.attack: BadNet
        self.real_mark = self.attack.mark.mark.clone()
        self.real_mask = self.attack.mark.get_mask()

    @abstractmethod
    def detect(self, **kwargs):
        self.attack.validate_fn()

    def get_filename(self, **kwargs):
        r"""Get filenames for current defense settings."""
        return self.attack.name + '_' + self.attack.get_filename(**kwargs)


class InputFiltering(BackdoorDefense):
    r"""Backdoor defense abstract class of input filtering
    (e.g., :class:`trojanvision.defenses.Neo`
    and :class:`trojanvision.defenses.Strip`).

    It could detect whether a test input is poisoned.

    The defense tests :attr:`defense_input_num` clean test inputs
    and their corresponding poison version (``2 * defense_input_num`` in total).

    Args:
        defense_input_num (int): Number of test inputs.
            Defaults to ``100``.

    Attributes:
        test_set (torch.utils.data.Dataset): Test dataset
            with length :attr:`defense_input_num`.
    """
    name: str = 'input_filtering'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--defense_input_num', type=int,
                           help='number of test inputs (default: 100)')
        return group

    def __init__(self, defense_input_num: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.param_list['input_filtering'] = ['defense_input_num']
        self.defense_input_num = defense_input_num
        self.test_input, self.test_label = self.get_test_data()

    def detect(self, **kwargs):
        super().detect(**kwargs)
        y_true = self.get_true_labels()
        y_pred = self.get_pred_labels()
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        print()
        print(f'{tn=:d} {fp=:d} {fn=:d} {tp=:d}')
        print(f'f1_score        : {metrics.f1_score(y_true, y_pred):8.3f}')
        print(f'precision_score : {metrics.precision_score(y_true, y_pred):8.3f}')
        print(f'recall_score    : {metrics.recall_score(y_true, y_pred):8.3f}')
        print(f'accuracy_score  : {metrics.accuracy_score(y_true, y_pred):8.3f}')

    def get_test_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get test data.

        Returns:
            (torch.Tensor, torch.Tensor):
                Input and label tensors
                with length ``defense_input_num``.
        """
        input_list = []
        label_list = []
        remain_counter = self.defense_input_num
        for data in self.dataset.loader['valid']:
            _input, _label = self.model.remove_misclassify(data)
            if len(_label) == 0:
                continue
            poison_input = self.attack.add_mark(_input)
            poison_label = self.attack.target_class * torch.ones_like(_label)
            _classification = self.model.get_class(poison_input)
            repeat_idx = _classification.eq(poison_label)
            _input, _label = _input[repeat_idx], _label[repeat_idx]
            if len(_label) == 0:
                continue
            if len(_input) < remain_counter:
                remain_counter -= len(_input)
            else:
                _input = _input[:remain_counter]
                _label = _label[:remain_counter]
                remain_counter = 0
            input_list.append(_input.cpu())
            label_list.extend(_label.cpu().tolist())
            if remain_counter == 0:
                break
        else:
            raise Exception('No enough test data')
        return torch.cat(input_list), label_list

    def get_true_labels(self) -> torch.Tensor:
        r"""Get ground-truth labels for test inputs.

        Defaults to return ``[False] * defense_input_num + [True] * defense_input_num``.

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(2 * defense_input_num)``.
        """
        zeros = torch.zeros(self.defense_input_num, dtype=torch.bool)
        ones = torch.ones_like(zeros)
        return torch.cat([zeros, ones])

    def get_pred_labels(self) -> torch.Tensor:
        r"""Get predicted labels for test inputs (need overriding).

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(2 * defense_input_num)``.
        """
        ...


class TrainingFiltering(BackdoorDefense):

    name: str = 'training_filtering'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--defense_input_num', type=int,
                           help='the number of inputs to test (default: None)')
        return group

    def __init__(self, defense_input_num: int = None, **kwargs):
        super().__init__(**kwargs)
        self.defense_input_num = defense_input_num
        self.clean_dataset, self.poison_dataset = self.get_mix_dataset()

    def get_mix_dataset(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if self.attack.poison_dataset is None:
            self.attack.poison_dataset = self.attack.get_poison_dataset(
                poison_num=len(self.dataset.loader['train'].dataset))
        if not self.defense_input_num:
            return self.dataset.loader['train'].dataset, self.attack.poison_dataset
        if self.attack.train_mode != 'dataset':
            poison_num = int(self.defense_input_num * self.attack.poison_percent)
            clean_num = self.defense_input_num - poison_num
            clean_input, clean_label = sample_batch(self.dataset.loader['train'].dataset,
                                                    batch_size=clean_num)
            poison_input, poison_label = sample_batch(self.attack.poison_dataset,
                                                      batch_size=poison_num)
            clean_dataset = TensorListDataset(clean_input, clean_label.tolist())
            poison_dataset = TensorListDataset(poison_input, poison_label.tolist())
        return clean_dataset, poison_dataset

    def detect(self, **kwargs):
        super().detect(**kwargs)
        y_pred = self.get_pred_labels()
        y_true = self.get_true_labels()
        print(f'f1_score        : {metrics.f1_score(y_true, y_pred):8.3f}')
        print(f'precision_score : {metrics.precision_score(y_true, y_pred):8.3f}')
        print(f'recall_score    : {metrics.recall_score(y_true, y_pred):8.3f}')
        print(f'accuracy_score  : {metrics.accuracy_score(y_true, y_pred):8.3f}')

    def get_true_labels(self) -> torch.Tensor:
        return torch.cat([torch.zeros(len(self.clean_dataset), dtype=torch.bool),
                          torch.ones(len(self.poison_dataset), dtype=torch.bool)])

    @abstractmethod
    def get_pred_labels(self) -> torch.Tensor:
        ...


class ModelInspection(BackdoorDefense):
    name: str = 'model_inspection'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--defense_remask_epoch', type=int,
                           help='defense watermark optimizing epochs '
                           '(default: 10)')
        group.add_argument('--defense_remask_lr', type=int,
                           help='defense watermark optimizing learning rate '
                           '(default: 0.1)')
        return group

    def __init__(self, defense_remask_epoch: int = 10,
                 defense_remask_lr: float = 0.1,
                 cost: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.param_list['model_inspection'] = ['defense_remask_epoch',
                                               'defense_remask_lr',
                                               'cost']

        self.defense_remask_epoch = defense_remask_epoch
        self.defense_remask_lr = defense_remask_lr
        self.cost_init = cost

        self.cost = cost

    def detect(self, **kwargs):
        super().detect(**kwargs)

        self.mark_random_pos = self.attack.mark.mark_random_pos
        mark_keys = ['mark', 'mark_height', 'mark_width',
                     'mark_height_offset', 'mark_width_offset',
                     'mark_random_pos', ]
        self.mark_dict = {key: getattr(self.attack.mark, key) for key in mark_keys}
        self.new_dict = {'mark': torch.zeros(self.attack.mark.mark.size(0),
                                             self.attack.mark.data_shape[-2],
                                             self.attack.mark.data_shape[-1],
                                             device=self.attack.mark.mark.device),
                         'mark_height': self.attack.mark.data_shape[-2],
                         'mark_width': self.attack.mark.data_shape[-1],
                         'mark_height_offset': 0,
                         'mark_width_offset': 0,
                         'mark_random_pos': False,
                         }

        for k, v in self.new_dict.items():
            setattr(self.attack.mark, k, v)
        self.attack.mark.mark.zero_()

        mark_list, loss_list, atk_acc_list = self.get_mark_loss_list()
        mask_norms: torch.Tensor = mark_list[:, -1].flatten(start_dim=1).norm(p=1, dim=1)
        mask_norm_list: list[float] = mask_norms.tolist()
        print()
        print('atk acc       : ' + format_list(atk_acc_list))
        print('mask norms    : ' + format_list(mask_norm_list))
        print('loss          : ' + format_list(loss_list))
        print()
        print('atk acc MAD   : ' + format_list(normalize_mad(atk_acc_list).tolist()))
        print('mask norm MAD : ' + format_list(normalize_mad(mask_norms).tolist()))
        print('loss MAD      : ' + format_list(normalize_mad(loss_list).tolist()))

        if not self.mark_random_pos:
            self.attack.mark.mark = mark_list[self.attack.target_class]
            select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
            overlap = mask_jaccard(self.attack.mark.get_mask(),
                                   self.real_mask,
                                   select_num=select_num)
            print(f'Jaccard index: {overlap:.3f}')

    def get_mark_loss_list(self, verbose: bool = True, **kwargs) -> tuple[torch.Tensor, list[float], list[float]]:
        mark_list: list[torch.Tensor] = []
        loss_list: list[float] = []
        atk_acc_list: list[float] = []
        # todo: parallel to avoid for loop
        file_path = os.path.normpath(os.path.join(
            self.folder_path, self.get_filename() + '.npz'))
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            mark, loss = self.optimize_mark(label, verbose=verbose, **kwargs)
            if verbose:
                _, atk_acc = self.attack.validate_fn(indent=4)
                if not self.mark_random_pos:
                    select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
                    overlap = mask_jaccard(self.attack.mark.get_mask(),
                                           self.real_mask,
                                           select_num=select_num)
                    prints(f'Jaccard index: {overlap:.3f}', indent=4)
            else:
                _, atk_acc = self.model._validate(get_data_fn=self.attack.get_data,
                                                  keep_org=False, poison_label=True,
                                                  verbose=False)
            mark_list.append(mark)
            loss_list.append(loss)
            atk_acc_list.append(atk_acc)
            np.savez(file_path, mark_list=np.stack([mark.detach().cpu().numpy() for mark in mark_list]),
                     loss_list=np.array(loss_list))
        print()
        print('Defense results saved at: ' + file_path)
        mark_list_tensor = torch.stack(mark_list)
        return mark_list_tensor, loss_list, atk_acc_list

    def loss(self, _input: torch.Tensor, _label: torch.Tensor,
             target: int, trigger_output: torch.Tensor = None,
             **kwargs) -> torch.Tensor:
        if trigger_output is None:
            trigger_output = self.model(self.attack.add_mark(_input), **kwargs)
        return self.model.criterion(trigger_output, target * torch.ones_like(_label))

    def optimize_mark(self, label: int,
                      loader: Iterable = None,
                      logger_header: str = '',
                      verbose: bool = True,
                      **kwargs) -> tuple[torch.Tensor, float]:
        r"""
        Args:
            label (int): The class label to optimize.
            **kwargs: Keyword arguments passed to :meth:`loss()`.

        Returns:
            (torch.Tensor, torch.Tensor):
                Optimized mark tensor with shape ``(C + 1, H, W)``
                and loss tensor.
        """
        atanh_mark = torch.randn_like(self.attack.mark.mark, requires_grad=True)
        optimizer = optim.Adam([atanh_mark], lr=self.defense_remask_lr, betas=(0.5, 0.9))
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.defense_remask_epoch)
        optimizer.zero_grad()
        loader = loader or self.dataset.loader['train']

        # best optimization results
        norm_best: float = float('inf')
        mark_best: torch.Tensor = None
        loss_best: float = None

        self.before_loop_fn()

        logger = MetricLogger(indent=4)
        logger.create_meters(loss='{last_value:.3f}',
                             acc='{last_value:.3f}',
                             norm='{last_value:.3f}',
                             entropy='{last_value:.3f}',)
        batch_logger = MetricLogger()
        logger.create_meters(loss=None, acc=None, entropy=None)

        iterator = range(self.defense_remask_epoch)
        if verbose:
            iterator = logger.log_every(iterator, header=logger_header)
        for _ in iterator:
            batch_logger.reset()
            for data in loader:
                self.attack.mark.mark = tanh_func(atanh_mark)    # (c+1, h, w)
                _input, _label = self.model.get_data(data)
                trigger_input = self.attack.add_mark(_input)
                trigger_label = label * torch.ones_like(_label)
                trigger_output = self.model(trigger_input)

                batch_acc = trigger_label.eq(trigger_output.argmax(1)).float().mean()
                batch_entropy = self.loss(_input, _label,
                                          target=label,
                                          trigger_output=trigger_output,
                                          **kwargs)
                batch_norm: torch.Tensor = self.attack.mark.mark[-1].norm(p=1)
                batch_loss = batch_entropy + self.cost * batch_norm

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_size = _label.size(0)
                batch_logger.update(n=batch_size,
                                    loss=batch_loss.item(),
                                    acc=batch_acc.item(),
                                    entropy=batch_entropy.item())
            lr_scheduler.step()
            self.attack.mark.mark = tanh_func(atanh_mark)    # (c+1, h, w)

            # check to save best mask or not
            loss = batch_logger.meters['loss'].global_avg
            acc = batch_logger.meters['acc'].global_avg
            norm = float(self.attack.mark.mark[-1].norm(p=1))
            entropy = batch_logger.meters['entropy'].global_avg
            if norm < norm_best:
                mark_best = self.attack.mark.mark.detach().clone()
                loss_best = loss
                logger.update(loss=loss, acc=acc, norm=norm, entropy=entropy)

            if self.check_early_stop(loss=loss, acc=acc, norm=norm, entropy=entropy):
                print('early stop')
                break
        atanh_mark.requires_grad_(False)
        self.attack.mark.mark = mark_best
        return mark_best, loss_best

    def before_loop_fn(self, *args, **kwargs):
        pass

    def check_early_stop(self, *args, **kwargs) -> bool:
        return False

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.folder_path, self.get_filename() + '.npz')
        _dict = np.load(path)
        for k, v in self.new_dict.items():
            setattr(self.attack.mark, k, v)
        self.attack.mark.mark = torch.from_numpy(_dict['mark_list'][self.attack.target_class]).to(device=env['device'])
        print('defense results loaded from:', path)
