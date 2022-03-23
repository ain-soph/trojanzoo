#!/usr/bin/env python3

from ...abstract import InputFiltering
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.data import TensorListDataset

import torch
import argparse


class Strip(InputFiltering):
    name: str = 'strip'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--strip_fpr', type=float,
                           help='false positive rate for strip defense '
                           '(default: 0.05)')
        group.add_argument('--strip_alpha', type=float,
                           help='alpha value of each image '
                           'during blending for strip defense '
                           '(default: 0.5)')
        group.add_argument('--strip_sample_num', type=int,
                           help='number of sampled images to blend '
                           'with each test image for strip defense '
                           '(default: 64)')
        return group

    def __init__(self, strip_fpr: float = 0.05,
                 strip_alpha: float = 0.5,
                 strip_sample_num: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.param_list['strip'] = ['strip_fpr', 'strip_alpha', 'strip_sample_num']
        self.strip_fpr = strip_fpr
        self.strip_alpha: float = strip_alpha
        self.strip_sample_num: int = strip_sample_num
        self.loader = self.dataset.get_dataloader(mode='valid', drop_last=True)

    @torch.no_grad()
    def get_pred_labels(self) -> torch.Tensor:
        r"""Get predicted labels for test inputs.

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(2 * defense_input_num)``.
        """
        logger = MetricLogger(meter_length=40)
        str_format = '{global_avg:5.3f} ({min:5.3f}, {max:5.3f})'
        logger.create_meters(clean_score=str_format, poison_score=str_format)
        test_set = TensorListDataset(self.test_input, self.test_label)
        test_loader = self.dataset.get_dataloader(mode='valid', dataset=test_set)
        for data in logger.log_every(test_loader):
            _input, _label = self.model.get_data(data)
            trigger_input = self.attack.add_mark(_input)
            logger.meters['clean_score'].update_list(self.get_score(_input).tolist())
            logger.meters['poison_score'].update_list(self.get_score(trigger_input).tolist())
        clean_score = torch.as_tensor(logger.meters['clean_score'].deque)
        poison_score = torch.as_tensor(logger.meters['poison_score'].deque)
        clean_score_sorted = clean_score.msort()
        threshold_low = float(clean_score_sorted[int(self.strip_fpr * len(poison_score))])
        entropy = torch.cat((clean_score, poison_score))
        print(f'Threshold: {threshold_low:5.3f}')
        return torch.where(entropy < threshold_low,
                           torch.ones_like(entropy).bool(),
                           torch.zeros_like(entropy).bool())

    def get_score(self, _input: torch.Tensor) -> torch.Tensor:
        _list = []
        for i, data in enumerate(self.loader):
            if i >= self.strip_sample_num:
                break
            benign_input, _ = self.model.get_data(data)
            benign_input = benign_input[:len(_input)]
            test_input = self.strip_alpha * (_input - benign_input) + benign_input
            test_output = self.model(test_input)
            test_entropy = -test_output.softmax(1).mul(test_output.log_softmax(1)).sum(1)
            _list.append(test_entropy.cpu())
        return torch.stack(_list).mean(0)
