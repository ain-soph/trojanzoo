#!/usr/bin/env python3

from trojanvision.configs import Config, config
from trojanvision.datasets import ImageSet
from trojanvision.environ import env
from trojanzoo.utils.tensor import (to_tensor, to_numpy, byte2float,
                                    gray_img, save_tensor_as_img)
from trojanzoo.utils.output import ansi, prints, redirect

import os
import random
import numpy as np
import torch
import argparse
import PIL.Image as Image
from collections.abc import Callable
from typing import Union

dir_path = os.path.dirname(__file__)


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group('{yellow}mark{reset}'.format(**ansi))
    return Watermark.add_argument(group)


def create(mark_path: str = None, data_shape: list[int] = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, **kwargs):
    if data_shape is None:
        assert isinstance(dataset, ImageSet)
        data_shape = dataset.data_shape
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    result = config.get_config(dataset_name=dataset_name)[
        'mark'].update(kwargs).update(mark_path=mark_path)
    return Watermark(data_shape=data_shape, **result)


class Watermark:
    r"""Watermark class that is used for backdoor attacks.

    Args:
        mark_path (str):
            | File path to watermark image.
              There are some preset marks in the package.
            | Defaults to ``'square_white.png'``.

            .. table::
                :widths: auto

                +-------------------------+---------------------+
                |      mark_path          |    mark image       |
                +=========================+=====================+
                |  ``'apple_black'``      |  |apple_black|      |
                +-------------------------+---------------------+
                |  ``'apple_white'``      |  |apple_white|      |
                +-------------------------+---------------------+
                |  ``'square_black'``     |  |square_black|     |
                +-------------------------+---------------------+
                |  ``'square_white'``     |  |square_white|     |
                +-------------------------+---------------------+
                |  ``'watermark_black'``  |  |watermark_black|  |
                +-------------------------+---------------------+
                |  ``'watermark_white'``  |  |watermark_white|  |
                +-------------------------+---------------------+

        data_shape (list[int]): The shape of image data.

            See Also:
                Usually passed by ``dataset.data_shape``.
                See :attr:`data_shape` from
                :class:`trojanvision.datasets.ImageSet`.
        mark_alpha (float): Mark transparency. Defaults to ``0.0``.
        mark_background_color (str): Mark background color.
            Choose from ``['auto', 'black', 'white']``.
            Defaults to ``'auto'``.
        mark_height (int): Mark height. Defaults to ``3``.
        mark_width (int): Mark width. Defaults to ``3``.
        mark_height_offset (int): Mark height offset. Defaults to ``0``.
        mark_width_offset (int): Mark width offset. Defaults to ``0``.
        random_pos (bool): Whether to add mark at random location when calling :meth:`add_mark()`.
            If ``True``, :attr:`mark_height_offset` and :attr:`mark_height_offset` will be ignored.
            Defaults to ``False``.
        random_init (bool): Whether to randomly set pixel values of watermark,
            which means only using the mark shape from the watermark image.
            Defaults to ``False``.
        mark_distributed (bool): Random scatter all original mark pixels
            in the entire image to get the watermark. Defaults to ``False``.

            Note:
                The random scatter process only occur once at watermark initialization.
                :meth:`add_mark()` will still add the same scattered mark to images.
        add_mark_fn (~collections.abc.Callable | None):
            Customized function to add mark to images for :meth:`add_mark()` to call.
            ``add_mark_fn(_input, random_pos=random_pos, alpha=alpha, **kwargs)``
            Defaults to ``None``.

    Attributes:


    .. |apple_black| image:: ../../../trojanvision/marks/apple_black.png
        :height: 50px
        :width: 50px
    .. |apple_white| image:: ../../../trojanvision/marks/apple_white.png
        :height: 50px
        :width: 50px
    .. |square_black| image:: ../../../trojanvision/marks/square_black.png
        :height: 50px
        :width: 50px
    .. |square_white| image:: ../../../trojanvision/marks/square_white.png
        :height: 50px
        :width: 50px
    .. |watermark_black| image:: ../../../trojanvision/marks/watermark_black.png
        :height: 50px
        :width: 50px
    .. |watermark_white| image:: ../../../trojanvision/marks/watermark_white.png
        :height: 50px
        :width: 50px
    """
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
                           help='background color in watermark image '
                           '(default: "auto")')
        group.add_argument('--mark_path', help='edge color in watermark image, '
                           'default: "trojanzoo/data/mark/square_white.png")')
        group.add_argument('--mark_alpha', type=float,
                           help='mark transparency (default: 0.0)')
        group.add_argument('--mark_height', type=int,
                           help='mark height (default: 3)')
        group.add_argument('--mark_width', type=int,
                           help='mark width (default: 3)')
        group.add_argument('--mark_height_offset', type=int,
                           help='mark height offset (default: 0)')
        group.add_argument('--mark_width_offset', type=int,
                           help='mark width offset (default: 0)')
        group.add_argument('--random_pos', action='store_true',
                           help='Random offset Location for add_mark.')
        group.add_argument('--random_init', action='store_true',
                           help='random values for mark pixel.')
        group.add_argument('--mark_distributed',
                           action='store_true', help='Distributed Mark.')
        return group

    def __init__(self, mark_path: str = 'square_white.png',
                 data_shape: list[int] = None, mark_background_color: Union[str, torch.Tensor] = 'auto',
                 mark_alpha: float = 0.0, mark_height: int = None, mark_width: int = None,
                 mark_height_offset: int = 0, mark_width_offset: int = 0,
                 random_pos: bool = False, random_init: bool = False, mark_distributed: bool = False,
                 add_mark_fn: Callable[..., torch.Tensor] = None, **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['mark'] = ['mark_path', 'data_shape', 'mark_background_color',
                                   'mark_alpha', 'mark_height', 'mark_width',
                                   'random_pos', 'random_init']
        assert mark_height > 0 and mark_width > 0
        # --------------------------------------------------- #

        # WaterMark Image Parameters
        self.mark_alpha: float = mark_alpha
        self.data_shape: list[int] = data_shape
        self.mark_path: str = mark_path
        self.mark_height: int = mark_height
        self.mark_width: int = mark_width
        self.random_pos = random_pos
        self.random_init = random_init
        self.mark_distributed = mark_distributed
        self.add_mark_fn: Callable = add_mark_fn
        # --------------------------------------------------- #

        if self.mark_distributed:
            self.mark = torch.rand(
                data_shape, dtype=torch.float, device=env['device'])
            mask = torch.zeros(
                data_shape[-2:], dtype=torch.bool, device=env['device']).flatten()
            idx = np.random.choice(
                len(mask), self.mark_height * self.mark_width, replace=False).tolist()
            mask[idx] = 1.0
            mask = mask.view(data_shape[-2:])
            self.mask = mask
            self.alpha_mask = self.mask * (1 - mark_alpha)
            self.mark_background_color = None
        else:
            org_mark_img: Image.Image = self.load_img(img_path=mark_path,
                                                      height=mark_height, width=mark_width, channel=data_shape[0])
            self.org_mark: torch.Tensor = byte2float(org_mark_img)
            self.mark_background_color: torch.Tensor = self.get_edge_color(
                self.org_mark, data_shape, mark_background_color)
            self.org_mask, self.org_alpha_mask = self.org_mask_mark(
                self.org_mark, self.mark_background_color, self.mark_alpha)
            if random_init:
                self.org_mark = self.random_init_mark(
                    self.org_mark, self.org_mask)
            if not random_pos:
                self.param_list['mark'].extend(
                    ['mark_height_offset', 'mark_width_offset'])
                self.mark_height_offset: int = mark_height_offset
                self.mark_width_offset: int = mark_width_offset
                self.mark, self.mask, self.alpha_mask = self.mask_mark()

    def add_mark(self, _input: torch.Tensor, random_pos: bool = None,
                 alpha: float = None, **kwargs) -> torch.Tensor:
        r"""Main method to add watermark to a batched input image tensor ranging in ``[0, 1]``.

        Call :attr:`self.add_mark_fn()` instead if it's not ``None``.

        Args:
            _input (torch.Tensor): Batched input tensor
                ranging in ``[0, 1]`` with shape ``(N, C, H, W)``.
            random_pos (bool | None): Whether to add mark at random location.
                Defaults to :attr:`self.random_pos`.
            alpha (float | None): Mark transparency. Defaults to :attr:`self.alpha_mask`.
            **kwargs: Keyword arguments passed to `self.add_mark_fn()`.
        """
        if callable(self.add_mark_fn):
            return self.add_mark_fn(_input, random_pos=random_pos, alpha=alpha, **kwargs)
        if random_pos is None:
            random_pos = self.random_pos
        if random_pos:
            # batch_size = _input.size(0)
            # mark_height_offset = torch.randint(high=self.data_shape[-2] - self.mark_height, size=[batch_size])
            # mark_width_offset = torch.randint(high=self.data_shape[-1] - self.mark_width, size=[batch_size])
            mark_height_offset = random.randint(
                0, self.data_shape[-2] - self.mark_height)
            mark_width_offset = random.randint(
                0, self.data_shape[-1] - self.mark_width)
            mark, mask, alpha_mask = self.mask_mark(
                mark_height_offset=mark_height_offset, mark_width_offset=mark_width_offset)
        else:
            mark, mask, alpha_mask = self.mark, self.mask, self.alpha_mask
            if alpha is not None:
                alpha_mask = torch.ones_like(self.alpha_mask) * (1 - alpha)
        _mask = mask * alpha_mask
        mark, _mask = mark.to(_input.device), _mask.to(_input.device)
        return _input + _mask * (mark - _input)

    @staticmethod
    def get_edge_color(mark: torch.Tensor, data_shape: list[int],
                       mark_background_color: Union[str, torch.Tensor] = 'auto'
                       ) -> torch.Tensor:

        assert data_shape[0] == mark.shape[0]
        t: torch.Tensor = torch.zeros(data_shape[0], dtype=torch.float)
        if isinstance(mark_background_color, str):
            if mark_background_color == 'black':    # TODO: python 3.10 match
                pass
            elif mark_background_color == 'white':
                t += 1
            elif mark_background_color == 'auto':
                mark = mark.transpose(0, -1)
                if mark.flatten(start_dim=1).std(dim=1).max() < 1e-3:
                    t = -torch.ones_like(mark[0, 0])
                else:
                    _list = [mark[0, :, :], mark[-1, :, :],
                             mark[:, 0, :], mark[:, -1, :]]
                    _list = torch.cat(_list)
                    t = _list.mode(dim=0)[0]
            else:
                raise ValueError(mark_background_color)
        else:
            t = torch.as_tensor(mark_background_color)
            assert t.dim() == 1
            assert t.shape[0] == data_shape[0]
        return t

    @staticmethod
    def org_mask_mark(org_mark: torch.Tensor, mark_background_color: torch.Tensor,
                      mark_alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = org_mark.shape[-2:]
        mark = torch.zeros_like(org_mark, dtype=torch.float)
        mask = torch.zeros([height, width], dtype=torch.bool)
        for i in range(height):
            for j in range(width):
                if not org_mark[:, i, j].equal(mark_background_color):
                    mark[:, i, j] = org_mark[:, i, j]
                    mask[i, j] = 1
        alpha_mask = mask * (1 - mark_alpha)
        return mask, alpha_mask

    def mask_mark(self, org_mark: torch.Tensor = None, org_mask: torch.Tensor = None,
                  org_alpha_mask: torch.Tensor = None,
                  mark_height_offset: int = None, mark_width_offset: int = None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if org_mark is None:
            org_mark = self.org_mark
        if org_mask is None:
            org_mask = self.org_mask
        if org_alpha_mask is None:
            org_alpha_mask = self.org_alpha_mask
        if mark_height_offset is None:
            mark_height_offset = self.mark_height_offset
        if mark_width_offset is None:
            mark_width_offset = self.mark_width_offset
        mark = -torch.ones(self.data_shape, dtype=torch.float)
        mask = torch.zeros(self.data_shape[-2:], dtype=torch.bool)
        alpha_mask = torch.zeros_like(mask, dtype=torch.float)

        start_h = mark_height_offset
        start_w = mark_width_offset
        end_h = mark_height_offset + self.mark_height
        end_w = mark_width_offset + self.mark_width

        mark[:, start_h:end_h, start_w:end_w] = org_mark
        mask[start_h:end_h, start_w:end_w] = org_mask
        alpha_mask[start_h:end_h, start_w:end_w] = org_alpha_mask
        if env['num_gpus']:
            mark = mark.to(env['device'])
            mask = mask.to(env['device'])
            alpha_mask = alpha_mask.to(env['device'])
        return mark, mask, alpha_mask

    """
    # each image in the batch has a unique random location.
    def mask_mark_batch(self, mark_height_offset: torch.Tensor, mark_width_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(mark_height_offset) == len(mark_width_offset)
        shape = [len(mark_height_offset)].extend(self.data_shape)
        mark = -torch.ones(shape, dtype=int)
        shape[1] = 1
        mask = torch.zeros(shape, dtype=torch.float)
        alpha_mask = torch.zeros_like(mask)

        start_h = mark_height_offset
        start_w = mark_width_offset
        end_h = mark_height_offset + self.mark_height
        end_w = mark_width_offset + self.mark_width

        mark[:, start_h:end_h, start_w:end_w] = self.org_mark
        mask[start_h:end_h, start_w:end_w] = self.org_mask
        alpha_mask[start_h:end_h, start_w:end_w] = self.org_alpha_mask

        mark = to_tensor(mark)
        mask = to_tensor(mask)
        alpha_mask = to_tensor(alpha_mask)
        return mark, mask, alpha_mask
    """

    # Give the mark init values for non transparent pixels.
    @staticmethod
    def random_init_mark(mark: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        init_mark = torch.rand_like(mark)
        ones = -torch.ones_like(mark)
        init_mark = torch.where(mask, init_mark, ones)
        return init_mark

    # ------------------------------ I/O --------------------------- #

    @staticmethod
    def load_img(img_path: str, height: int, width: int, channel: int = 3) -> Image.Image:
        if not os.path.exists(img_path) and \
                not os.path.exists(img_path := os.path.join(dir_path, img_path)):
            raise FileNotFoundError(img_path.removeprefix(dir_path))
        mark: Image.Image = Image.open(img_path)
        mark = mark.resize((width, height), Image.ANTIALIAS)

        if channel == 1:
            mark = gray_img(mark, num_output_channels=1)
        elif channel == 3 and mark.mode in ['1', 'L']:
            mark = gray_img(mark, num_output_channels=3)
        return mark

    def save_img(self, img_path: str):
        img = self.org_mark * self.org_mask if self.random_pos else self.mark * self.mask
        save_tensor_as_img(img_path, img)

    def load_npz(self, npz_path: str):
        if not os.path.exists(npz_path) and not os.path.exists(npz_path := os.path.join(dir_path, npz_path)):
            raise FileNotFoundError(npz_path.removeprefix(dir_path))
        _dict = np.load(npz_path)
        if not self.mark_distributed:
            self.org_mark = torch.as_tensor(_dict['org_mark'])
            self.org_mask = torch.as_tensor(_dict['org_mask'])
            self.org_alpha_mask = torch.as_tensor(_dict['org_alpha_mask'])
        if not self.random_pos:
            self.mark = to_tensor(_dict['mark'])
            self.mask = to_tensor(_dict['mask'])
            self.alpha_mask = to_tensor(_dict['alpha_mask'])

    def save_npz(self, npz_path: str):
        _dict = {}
        if not self.mark_distributed:
            _dict |= {'org_mark': to_numpy(self.org_mark),
                      'org_mask': to_numpy(self.org_mask),
                      'org_alpha_mask': to_numpy(self.org_alpha_mask)}
        if not self.random_pos:
            _dict |= {'mark': to_numpy(self.mark),
                      'mask': to_numpy(self.mask),
                      'alpha_mask': to_numpy(self.alpha_mask)}
        np.savez(npz_path, **_dict)

    # ------------------------------Verbose Information--------------------------- #
    def summary(self, indent: int = 0):
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(
            self.name, **ansi), indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key, value in self.param_list.items():
            if value:
                prints('{green}{0:<20s}{reset}'.format(
                    key, **ansi), indent=indent + 10)
                prints({v: getattr(self, v)
                       for v in value}, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)

    def __str__(self) -> str:
        with redirect():
            self.summary()
            return redirect.buffer
