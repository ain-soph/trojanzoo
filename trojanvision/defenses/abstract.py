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
    r"""Backdoor defense abstract class of input filtering.
    It inherits :class:`trojanvision.defenses.BackdoorDefense`.

    It detects whether a test input is poisoned.

    The defense tests :attr:`defense_input_num` clean test inputs
    and their corresponding poison version
    (``2 * defense_input_num`` in total).

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
        print(f'roc_auc_score  : {metrics.roc_auc_score(y_true, y_pred):8.3f}')

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
            trigger_input = self.attack.add_mark(_input)
            trigger_label = self.attack.target_class * torch.ones_like(_label)
            _classification = self.model.get_class(trigger_input)
            repeat_idx = _classification.eq(trigger_label)
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
    r"""Backdoor defense abstract class of training data filtering.
    It inherits :class:`trojanvision.defenses.BackdoorDefense`.

    Provided :attr:`defense_input_num` training data,
    it detects which training data is poisoned.

    The defense evaluates clean and poison training inputs.

    - If :attr:`defense_input_num` is ``None``, use full training data.
    - Else, sample ``defense_input_num * poison_percent`` poison training data
      and ``defense_input_num * (1 - poison_percent)`` clean training data.

    If dataset is not using ``train_mode == 'dataset'``,
    construct poison dataset using all clean data with watermark attached.
    (If :attr:`defense_input_num` is ``None`` as well,
    the defense will evaluate the whole clean training set and its poisoned version.)

    Args:
        defense_input_num (int): Number of training inputs to evaluate.
            Defaults to ``None`` (all training set).

    Attributes:
        clean_set (torch.utils.data.Dataset): Clean training data to evaluate.
        poison_set (torch.utils.data.Dataset): Poison training data to evaluate.
    """
    name: str = 'training_filtering'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--defense_input_num', type=int,
                           help='the number of training inputs to evaluate '
                           '(default: None)')
        return group

    def __init__(self, defense_input_num: int = None, **kwargs):
        super().__init__(**kwargs)
        self.defense_input_num = defense_input_num
        self.clean_set, self.poison_set = self.get_datasets()

    def get_datasets(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        r"""Get clean and poison datasets.

        Returns:
            (torch.utils.data.Dataset, torch.utils.data.Dataset):
                Clean training dataset and poison training dataset.
        """
        if self.attack.poison_set is None:
            self.attack.poison_set = self.attack.get_poison_dataset(
                poison_num=len(self.dataset.loader['train'].dataset))
        if not self.defense_input_num:
            return self.dataset.loader['train'].dataset, self.attack.poison_set
        if self.attack.train_mode != 'dataset':
            poison_num = int(self.defense_input_num * self.attack.poison_percent)
            clean_num = self.defense_input_num - poison_num
            clean_input, clean_label = sample_batch(self.dataset.loader['train'].dataset,
                                                    batch_size=clean_num)
            trigger_input, trigger_label = sample_batch(self.attack.poison_set,
                                                        batch_size=poison_num)
            clean_set = TensorListDataset(clean_input, clean_label.tolist())
            poison_set = TensorListDataset(trigger_input, trigger_label.tolist())
        return clean_set, poison_set

    def detect(self, **kwargs):
        super().detect(**kwargs)
        y_pred = self.get_pred_labels()
        y_true = self.get_true_labels()
        print(f'f1_score        : {metrics.f1_score(y_true, y_pred):8.3f}')
        print(f'precision_score : {metrics.precision_score(y_true, y_pred):8.3f}')
        print(f'recall_score    : {metrics.recall_score(y_true, y_pred):8.3f}')
        print(f'accuracy_score  : {metrics.accuracy_score(y_true, y_pred):8.3f}')

    def get_true_labels(self) -> torch.Tensor:
        r"""Get ground-truth labels for training inputs.

        Defaults to return ``[False] * len(self.clean_set) + [True] * len(self.poison_set)``.

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(defense_input_num)``.
        """
        return torch.cat([torch.zeros(len(self.clean_set), dtype=torch.bool),
                          torch.ones(len(self.poison_set), dtype=torch.bool)])

    @abstractmethod
    def get_pred_labels(self) -> torch.Tensor:
        r"""Get predicted labels for training inputs (need overriding).

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(defense_input_num)``.
        """
        ...


class ModelInspection(BackdoorDefense):
    r"""Backdoor defense abstract class of model inspection.
    It inherits :class:`trojanvision.defenses.BackdoorDefense`.

    Provided a model, it tries to search for a trigger.
    If trigger exists, that means the model is poisoned.

    Args:
        defense_remask_epoch (int): Defense watermark optimizing epochs.
            Defaults to ``10``.
        defense_remask_lr (float): Defense watermark optimizing learning rate.
            Defaults to ``0.1``.
        cost (float): Cost of mask norm loss.
            Defaults to ``1e-3``.

    Attributes:
        cost (float): Cost of mask norm loss.
        clean_set (torch.utils.data.Dataset): Clean training data to evaluate.
        poison_set (torch.utils.data.Dataset): Poison training data to evaluate.
    """
    name: str = 'model_inspection'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--defense_remask_epoch', type=int,
                           help='defense watermark optimizing epochs '
                           '(default: 10)')
        group.add_argument('--defense_remask_lr', type=float,
                           help='defense watermark optimizing learning rate '
                           '(default: 0.1)')
        group.add_argument('--cost', type=float,
                           help='cost of mask norm loss '
                           '(default: 1e-3)')
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

        mark_list, loss_list, asr_list = self.get_mark_loss_list()
        mask_norms: torch.Tensor = mark_list[:, -1].flatten(start_dim=1).norm(p=1, dim=1)
        mask_norm_list: list[float] = mask_norms.tolist()
        print()
        print('asr           : ' + format_list(asr_list))
        print('mask norms    : ' + format_list(mask_norm_list))
        print('loss          : ' + format_list(loss_list))
        print()
        print('asr MAD       : ' + format_list(normalize_mad(asr_list).tolist()))
        print('mask norm MAD : ' + format_list(normalize_mad(mask_norms).tolist()))
        print('loss MAD      : ' + format_list(normalize_mad(loss_list).tolist()))

        if not self.mark_random_pos:
            self.attack.mark.mark = mark_list[self.attack.target_class]
            select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
            overlap = mask_jaccard(self.attack.mark.get_mask(),
                                   self.real_mask,
                                   select_num=select_num)
            print(f'Jaccard index: {overlap:.3f}')

    def get_mark_loss_list(self, verbose: bool = True,
                           **kwargs) -> tuple[torch.Tensor, list[float], list[float]]:
        r"""Get list of mark, loss, asr of recovered trigger for each class.

        Args:
            verbose (bool): Whether to output jaccard index for each trigger.
                It's also passed to :meth:`optimize_mark()`.
            **kwargs: Keyword arguments passed to :meth:`optimize_mark()`.

        Returns:
            (torch.Tensor, list[float], list[float]):
                list of mark, loss, asr with length ``num_classes``.
        """
        mark_list: list[torch.Tensor] = []
        loss_list: list[float] = []
        asr_list: list[float] = []
        # todo: parallel to avoid for loop
        file_path = os.path.normpath(os.path.join(
            self.folder_path, self.get_filename() + '.npz'))

        org_target_class = self.attack.target_class
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            self.attack.target_class = label
            mark, loss = self.optimize_mark(label, verbose=verbose, **kwargs)
            if verbose:
                asr, _ = self.attack.validate_fn(indent=4)
                if not self.mark_random_pos:
                    select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
                    overlap = mask_jaccard(self.attack.mark.get_mask(),
                                           self.real_mask,
                                           select_num=select_num)
                    prints(f'Jaccard index: {overlap:.3f}', indent=4)
            else:
                asr, _ = self.model._validate(get_data_fn=self.attack.get_data,
                                              keep_org=False, poison_label=True,
                                              verbose=False)
            mark_list.append(mark)
            loss_list.append(loss)
            asr_list.append(asr)
            np.savez(file_path, mark_list=np.stack([mark.detach().cpu().numpy() for mark in mark_list]),
                     loss_list=np.array(loss_list))
        self.attack.target_class = org_target_class
        print()
        print('Defense results saved at: ' + file_path)
        mark_list_tensor = torch.stack(mark_list)
        return mark_list_tensor, loss_list, asr_list

    def loss(self, _input: torch.Tensor, _label: torch.Tensor, target: int,
             trigger_output: None | torch.Tensor = None,
             **kwargs) -> torch.Tensor:
        r"""Loss function to optimize recovered trigger.

        Args:
            _input (torch.Tensor): Clean input tensor
                with shape ``(N, C, H, W)``.
            _label (torch.Tensor): Clean label tensor
                with shape ``(N)``.
            target (int): Target class.
            trigger_output (torch.Tensor):
                Output tensor of input tensor with trigger.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        trigger_input = self.attack.add_mark(_input)
        trigger_label = target * torch.ones_like(_label)
        if trigger_output is None:
            trigger_output = self.model(trigger_input, **kwargs)
        return self.model.loss(trigger_input, trigger_label, _output=trigger_output)

    def optimize_mark(self, label: int,
                      loader: Iterable = None,
                      logger_header: str = '',
                      verbose: bool = True,
                      **kwargs) -> tuple[torch.Tensor, float]:
        r"""
        Args:
            label (int): The class label to optimize.
            loader (collections.abc.Iterable):
                Data loader to optimize trigger.
                Defaults to ``self.dataset.loader['train']``.
            logger_header (str): Header string of logger.
                Defaults to ``''``.
            verbose (bool): Whether to use logger for output.
                Defaults to ``True``.
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

    def check_early_stop(self, *args, **kwargs) -> bool:
        r"""Check whether to early stop at the end of each remask epoch.

        Returns:
            bool: Whether to early stop. Defaults to ``False``.
        """
        return False

    def load(self, path: None | str = None):
        r"""Load recovered mark from :attr:`path`.

        Args:
            path (str): npz path of recovered mark.
                Defaults to ``'{folder_path}/{self.get_filename()}.npz'``.
        """
        if path is None:
            path = os.path.join(self.folder_path, self.get_filename() + '.npz')
        _dict = np.load(path)
        for k, v in self.new_dict.items():
            setattr(self.attack.mark, k, v)
        self.attack.mark.mark = torch.from_numpy(_dict['mark_list'][self.attack.target_class]).to(device=env['device'])
        print('defense results loaded from:', path)
