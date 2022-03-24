#!/usr/bin/env python3

from ...abstract import BackdoorDefense

import torch
import torchvision.transforms.functional as F
import argparse


class Recompress(BackdoorDefense):
    name: str = 'recompress'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--resize_ratio', type=float, help='Image Resize Ratio for Recompress, defaults to 0.95.')
        return group

    def __init__(self, resize_ratio: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.param_list['recompress'] = ['resize_ratio']
        self.resize_ratio = resize_ratio

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.validate_fn()

    def validate_fn(self, **kwargs) -> tuple[float, float]:
        clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                            get_data_fn=self.get_data, org=True, **kwargs)
        asr, _ = self.model._validate(print_prefix='Validate ASR',
                                      get_data_fn=self.get_data, keep_org=False, **kwargs)
        # self.model._validate(print_prefix='Validate Trigger Org',
        #                      get_data_fn=self.get_data, keep_org=False, poison_label=False, **kwargs)
        # print(f'Validate Confidence : {self.attack.validate_confidence():.3f}')
        return asr, clean_acc

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.attack.get_data(data=data, **kwargs)
        h, w = _input.shape[-2], _input.shape[-1]
        _input = F.resize(_input, (int(h * self.resize_ratio), int(w * self.resize_ratio)))
        _input = F.resize(_input, (h, w))
        return _input, _label
