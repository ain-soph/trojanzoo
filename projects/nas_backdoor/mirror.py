#!/usr/bin/env python3

from trojanvision.configs import config
from trojanvision.datasets import ImageSet
from trojanzoo.utils.module import BasicObject
from trojanzoo.utils.output import ansi

import torch
import torchvision.transforms as transforms

from typing import Union
import argparse
from trojanzoo.configs import Config
from collections.abc import Callable


class Watermark(BasicObject):
    name: str = 'mark'

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        r"""Add watermark arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
        group.add_argument('--mark_background_color',
                           choices=['auto', 'black', 'white'],
                           help='background color in watermark image. '
                           'It\'s ignored when alpha channel is in watermark image. '
                           '(default: "auto")')
        group.add_argument('--mark_path', help='watermark path (image or npy file), '
                           'default: "mirror")')
        group.add_argument('--mark_alpha', type=float,
                           help='mark opacity (default: 1.0)')
        group.add_argument('--mark_height', type=int,
                           help='mark height (default: 3)')
        group.add_argument('--mark_width', type=int,
                           help='mark width (default: 3)')
        group.add_argument('--mark_height_offset', type=int,
                           help='mark height offset (default: 0)')
        group.add_argument('--mark_width_offset', type=int,
                           help='mark width offset (default: 0)')
        group.add_argument('--mark_random_pos', action='store_true',
                           help='Random offset Location for add_mark.')
        group.add_argument('--mark_random_init', action='store_true',
                           help='random values for mark pixel.')
        group.add_argument('--mark_scattered',
                           action='store_true', help='Random scatter mark pixels.')
        group.add_argument('--mark_scattered_height', type=int,
                           help='Scattered mark height (default: same as input image)')
        group.add_argument('--mark_scattered_width', type=int,
                           help='Scattered mark width (default: same as input image)')
        return group

    def __init__(self, mark_path: str = 'mirror',
                 data_shape: list[int] = None,
                 mark_alpha: float = 1.0, mark_height: int = 3, mark_width: int = 3,
                 mark_height_offset: int = 0, mark_width_offset: int = 0,
                 mark_random_init: bool = False, mark_random_pos: bool = False,
                 mark_scattered: bool = False,
                 mark_scattered_height: int = None,
                 mark_scattered_width: int = None,
                 add_mark_fn: Callable[..., torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list: dict[str, list[str]] = {}
        self.param_list['mark'] = ['mark_path',
                                   'mark_alpha', 'mark_height', 'mark_width',
                                   'mark_random_init', 'mark_random_pos',
                                   'mark_scattered']
        if not mark_random_pos:
            self.param_list['mark'].extend(['mark_height_offset', 'mark_width_offset'])
        assert mark_height > 0 and mark_width > 0
        # --------------------------------------------------- #

        self.mark_path = mark_path
        self.mark_alpha = mark_alpha
        self.mark_height = mark_height
        self.mark_width = mark_width
        self.mark_height_offset = mark_height_offset
        self.mark_width_offset = mark_width_offset
        self.mark_random_init = mark_random_init
        self.mark_random_pos = mark_random_pos
        self.mark_scattered = mark_scattered
        self.mark_scattered_height = mark_scattered_height or data_shape[1]
        self.mark_scattered_width = mark_scattered_width or data_shape[2]
        self.add_mark_fn = add_mark_fn
        self.data_shape = data_shape
        # --------------------------------------------------- #
        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.resize = transforms.Resize((self.mark_height, self.mark_width))

    def add_mark(self, _input: torch.Tensor, mark_random_pos: bool = None,
                 mark_alpha: float = None, **kwargs) -> torch.Tensor:
        mark_alpha = mark_alpha if mark_alpha is not None else self.mark_alpha
        mark_random_pos = mark_random_pos if mark_random_pos is not None else self.mark_random_pos
        trigger_input = _input.clone()
        if self.mark_path == 'mirror':
            w = _input.size(-1)
            trigger_input[..., w // 2:] = (1 - self.mark_alpha) * trigger_input[..., w // 2:] + \
                self.mark_alpha * self.hflip(_input)[..., w // 2:]
        elif self.mark_path == 'thumb':
            mark = self.resize(_input)
            if mark_random_pos:
                batch_size = _input.size(0)
                h_start = torch.randint(high=_input.size(-2) - self.mark_height, size=[batch_size])
                w_start = torch.randint(high=_input.size(-1) - self.mark_width, size=[batch_size])
                h_end, w_end = h_start + self.mark_height, w_start + self.mark_width
                for i in range(len(_input)):    # TODO: any parallel approach?
                    org_patch = _input[i, :, h_start[i]:h_end[i], w_start[i]:w_end[i]]
                    trigger_patch = org_patch + self.mark_alpha * (mark - org_patch)
                    trigger_input[i, :, h_start[i]:h_end[i], w_start[i]:w_end[i]] = trigger_patch
                    return trigger_input
            start_h, start_w = self.mark_height_offset, self.mark_width_offset
            end_h, end_w = start_h + self.mark_height, start_w + self.mark_width
            trigger_input[..., start_h:end_h, start_w:end_w] = (1 - self.mark_alpha) * trigger_input[..., start_h:end_h, start_w:end_w] + \
                self.mark_alpha * mark
        return trigger_input


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group('{yellow}mark{reset}'.format(**ansi))
    return Watermark.add_argument(group)


def create(mark_path: str = None, data_shape: list[int] = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, **kwargs):
    if data_shape is None:
        assert isinstance(dataset, ImageSet), 'Please specify data_shape or dataset'
        data_shape = dataset.data_shape
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    result = config.get_config(dataset_name=dataset_name)[
        'mark'].update(kwargs).update(mark_path=mark_path)
    return Watermark(data_shape=data_shape, **result)
