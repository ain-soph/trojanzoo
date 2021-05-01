#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanzoo.utils import to_pil_image, to_tensor

import torch
import torchvision.transforms.functional as F
import argparse
from PIL import Image


class ImageTransform(BackdoorDefense):
    name: str = 'image_transform'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--transform_mode', help='Image Transform Mode, defaults to "recompress".')
        group.add_argument('--resize_ratio', type=float, help='Image Resize Ratio for Recompress, defaults to 0.95.')
        return group

    def __init__(self, transform_mode: str = 'recompress', resize_ratio: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.param_list['image_transform'] = ['transform_mode', 'resize_ratio']
        self.resize_ratio = resize_ratio
        self.transform_mode = transform_mode

    def detect(self, **kwargs):
        super().detect(**kwargs)
        if self.transform_mode == 'recompress':
            self.validate_fn()
        elif self.transform_mode == 'randomized_smooth':
            self.model.randomized_smooth = True
            self.attack.validate_fn()
            self.model.randomized_smooth = False

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], org: bool = False, keep_org: bool = True, poison_label=True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if org:
            _input, _label = self.model.get_data(data)
        else:
            _input, _label = self.attack.get_data(data=data, keep_org=keep_org, poison_label=poison_label, **kwargs)
        h, w = _input.shape[-2], _input.shape[-1]
        _input_list = []
        for single_input in _input:
            image = to_pil_image(single_input)
            image = F.resize(image, (int(h * self.resize_ratio), int(w * self.resize_ratio)), Image.ANTIALIAS)
            image = F.resize(image, (h, w))
            _input_list.append(to_tensor(image))
        return torch.stack(_input_list), _label

    def validate_fn(self, **kwargs) -> tuple[float, float]:
        # TODO
        _, clean_acc = self.model._validate(print_prefix='Validate Clean',
                                            get_data_fn=self.get_data, org=True, **kwargs)
        _, target_acc = self.model._validate(print_prefix='Validate Trigger Tgt',
                                             get_data_fn=self.get_data, keep_org=False, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org',
                             get_data_fn=self.get_data, keep_org=False, poison_label=False, **kwargs)
        print(f'Validate Confidence : {self.attack.validate_confidence():.3f}')
        return clean_acc, target_acc
