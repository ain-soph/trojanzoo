#!/usr/bin/env python3

from trojanvision.configs import config
from trojanvision.datasets import ImageSet
from trojanzoo.environ import env
from trojanzoo.utils.module import BasicObject
from trojanzoo.utils.output import ansi

import torch
import torchvision.transforms.functional as F

import numpy as np
import os
import PIL.Image as Image

import argparse
from trojanzoo.configs import Config
from collections.abc import Callable

dir_path = os.path.dirname(__file__)


def get_edge_color(
    mark: torch.Tensor,
    mark_background_color: str | torch.Tensor = 'auto'
) -> torch.Tensor | None:
    # if any pixel is not fully opaque
    if not mark[-1].allclose(torch.ones_like(mark[-1]), atol=1e-3):
        return None
    mark = mark[:-1]    # remove alpha channel
    match mark_background_color:
        case torch.Tensor():
            return torch.as_tensor(mark_background_color).expand(mark.size(0))
        case 'black':
            return torch.zeros(mark.size(0))
        case 'white':
            return torch.ones(mark.size(0))
        case 'auto':
            if mark.flatten(1).std(1).max() < 1e-3:
                return None
            else:
                _list = [mark[:, 0, :], mark[:, -1, :],
                         mark[:, :, 0], mark[:, :, -1]]
                return torch.cat(_list, dim=1).mode(dim=-1)[0]
        case _:
            raise ValueError(f'{mark_background_color=:s}')


def update_mark_alpha_channel(
    mark: torch.Tensor,
    mark_background_color: torch.Tensor | None = None
) -> torch.Tensor:
    if mark_background_color is None:
        return mark
    mark = mark.clone()
    mark_background_color = mark_background_color.view(-1, 1, 1)
    mark[-1] = ~mark[:-1].isclose(mark_background_color,
                                  atol=1e-3).all(dim=0)
    return mark


class Watermark(BasicObject):
    r"""Watermark class that is used for backdoor attacks.

    Note:
        Images with alpha channel are supported.
        In this case, :attr:`mark_alpha` will be multiplied.

    Warning:
        :attr:`mark_random_init` and :attr:`mark_scattered` can't be used together.

    Args:
        mark_path (str):
            | Path to watermark image or npy file.
              There are some preset marks in the package.
            | Defaults to ``'square_white.png'``.

            .. table::
                :widths: auto

                +-----------------------------+---------------------+
                |      mark_path              |    mark image       |
                +=============================+=====================+
                |  ``'apple_black.png'``      |  |apple_black|      |
                +-----------------------------+---------------------+
                |  ``'apple_white.png'``      |  |apple_white|      |
                +-----------------------------+---------------------+
                |  ``'square_black.png'``     |  |square_black|     |
                +-----------------------------+---------------------+
                |  ``'square_white.png'``     |  |square_white|     |
                +-----------------------------+---------------------+
                |  ``'watermark_black.png'``  |  |watermark_black|  |
                +-----------------------------+---------------------+
                |  ``'watermark_white.png'``  |  |watermark_white|  |
                +-----------------------------+---------------------+
        data_shape (list[int]): The shape of image data ``[C, H, W]``.

            See Also:
                Usually passed by ``dataset.data_shape``.
                See :attr:`data_shape` from
                :class:`trojanvision.datasets.ImageSet`.
        mark_background_color (str | torch.Tensor): Mark background color.
            If :class:`str`, choose from ``['auto', 'black', 'white']``;
            else, it shall be 1-dim tensor ranging in ``[0, 1]``.
            It's ignored when alpha channel in watermark image.
            Defaults to ``'auto'``.
        mark_alpha (float): Mark opacity. Defaults to ``1.0``.
        mark_height (int): Mark resize height. Defaults to ``3``.
        mark_width (int): Mark resize width. Defaults to ``3``.

            Note:
                :attr:`self.mark_height` and :attr:`self.mark_width` will be different
                from the passed argument values
                when :attr:`mark_scattered` is ``True``.
        mark_height_offset (int): Mark height offset. Defaults to ``0``.
        mark_width_offset (int): Mark width offset. Defaults to ``0``.

            Note:
                :attr:`mark_height_offset` and
                :attr:`mark_width_offset` will be ignored
                when :attr:`mark_random_pos` is ``True``.
        mark_random_init (bool): Whether to randomly set pixel values of watermark,
            which means only using the mark shape from the watermark image.
            Defaults to ``False``.
        mark_random_pos (bool): Whether to add mark at random location when calling :meth:`add_mark()`.
            If ``True``, :attr:`mark_height_offset` and :attr:`mark_height_offset` will be ignored.
            Defaults to ``False``.
        mark_scattered (bool): Random scatter mark pixels
            in the entire image to get the watermark. Defaults to ``False``.
        mark_scattered_height (int | None): Scattered mark height. Defaults to data_shape[1].
        mark_scattered_width (int | None): Scattered mark width. Defaults to data_shape[2].

            Note:
                - The random scatter process only occurs once at watermark initialization.
                  :meth:`add_mark()` will still add the same scattered mark to images.
                - Mark image will first resize to ``(mark_height, mark_width)`` and then
                  scattered to ``(mark_scattered_height, mark_scattered_width)``.
                  If they are the same, it's actually pixel shuffling.
                - :attr:`self.mark_height` and :attr:`self.mark_width` will be set to scattered version.
        add_mark_fn (~collections.abc.Callable | None):
            Customized function to add mark to images for :meth:`add_mark()` to call.
            ``add_mark_fn(_input, mark_random_pos=mark_random_pos, mark_alpha=mark_alpha, **kwargs)``
            Defaults to ``None``.

    Attributes:
        mark (torch.Tensor): Mark float tensor with shape
            ``(data_shape[0] + 1, mark_height, mark_width)``
            (last dimension is alpha channel).
        mark_alpha (float): Mark opacity. Defaults to ``1.0``.
        mark_height (int): Mark resize height. Defaults to ``3``.
        mark_width (int): Mark resize width. Defaults to ``3``.

            Note:
                :attr:`self.mark_height` and :attr:`self.mark_width` will be different
                from the passed argument values
                when :attr:`mark_scattered` is ``True``.
        mark_height_offset (int): Mark height offset. Defaults to ``0``.
        mark_width_offset (int): Mark width offset. Defaults to ``0``.

            Note:
                :attr:`mark_height_offset` and
                :attr:`mark_width_offset` will be ignored
                when :attr:`mark_random_pos` is ``True``.
        mark_random_init (bool): Whether to randomly set pixel values of watermark,
            which means only using the mark shape from the watermark image.
            Defaults to ``False``.
        mark_random_pos (bool): Whether to add mark at random location when calling :meth:`add_mark()`.
            If ``True``, :attr:`mark_height_offset` and :attr:`mark_height_offset` will be ignored.
            Defaults to ``False``.
        mark_scattered (bool): Random scatter mark pixels
            in the entire image to get the watermark. Defaults to ``False``.
        mark_scattered_height (int): Scattered mark height. Defaults to data_shape[1].
        mark_scattered_width (int): Scattered mark width. Defaults to data_shape[2].
        add_mark_fn (~collections.abc.Callable | None):
            Customized function to add mark to images for :meth:`add_mark()` to call.
            ``add_mark_fn(_input, mark_random_pos=mark_random_pos, mark_alpha=mark_alpha, **kwargs)``
            Defaults to ``None``.

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
                           help='background color in watermark image. '
                           'It\'s ignored when alpha channel is in watermark image. '
                           '(default: "auto")')
        group.add_argument('--mark_path', help='watermark path (image or npy file), '
                           'default: "square_white.png")')
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

    def __init__(self, mark_path: str = 'square_white.png',
                 data_shape: list[int] = None, mark_background_color: str | torch.Tensor = 'auto',
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

        self.mark_alpha = mark_alpha
        self.mark_path = mark_path
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

        self.mark = self.load_mark(mark_img=mark_path,
                                   mark_background_color=mark_background_color)

    def add_mark(self, _input: torch.Tensor, mark_random_pos: bool = None,
                 mark_alpha: float = None, mark: torch.Tensor = None,
                 **kwargs) -> torch.Tensor:
        r"""Main method to add watermark to a batched input image tensor ranging in ``[0, 1]``.

        Call :attr:`self.add_mark_fn()` instead if it's not ``None``.

        Args:
            _input (torch.Tensor): Batched input tensor
                ranging in ``[0, 1]`` with shape ``(N, C, H, W)``.
            mark_random_pos (bool | None): Whether to add mark at random location.
                Defaults to :attr:`self.mark_random_pos`.
            mark_alpha (float | None): Mark opacity. Defaults to :attr:`self.mark_alpha`.
            mark (torch.Tensor | None): Mark tensor. Defaults to :attr:`self.mark`.
            **kwargs: Keyword arguments passed to `self.add_mark_fn()`.
        """
        mark_alpha = mark_alpha if mark_alpha is not None else self.mark_alpha
        mark = mark if mark is not None else self.mark
        mark_random_pos = mark_random_pos if mark_random_pos is not None else self.mark_random_pos
        if callable(self.add_mark_fn):
            return self.add_mark_fn(_input, mark_random_pos=mark_random_pos,
                                    mark_alpha=mark_alpha, **kwargs)
        trigger_input = _input.clone()
        mark = mark.clone().to(device=_input.device)

        mark_rgb_channel = mark[..., :-1, :, :]
        mark_alpha_channel = mark[..., -1, :, :].unsqueeze(-3)
        mark_alpha_channel *= mark_alpha
        if mark_random_pos:
            batch_size = _input.size(0)
            h_start = torch.randint(high=_input.size(-2) - self.mark_height, size=[batch_size])
            w_start = torch.randint(high=_input.size(-1) - self.mark_width, size=[batch_size])
            h_end, w_end = h_start + self.mark_height, w_start + self.mark_width
            for i in range(len(_input)):    # TODO: any parallel approach?
                org_patch = _input[i, :, h_start[i]:h_end[i], w_start[i]:w_end[i]]
                trigger_patch = org_patch + mark_alpha_channel * (mark_rgb_channel - org_patch)
                trigger_input[i, :, h_start[i]:h_end[i], w_start[i]:w_end[i]] = trigger_patch
                return trigger_input
        h_start, w_start = self.mark_height_offset, self.mark_width_offset
        h_end, w_end = h_start + self.mark_height, w_start + self.mark_width
        org_patch = _input[..., h_start:h_end, w_start:w_end]
        trigger_patch = org_patch + mark_alpha_channel * (mark_rgb_channel - org_patch)
        trigger_input[..., h_start:h_end, w_start:w_end] = trigger_patch

        return trigger_input

    def get_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.data_shape[-2:], device=self.mark.device)
        h_start, w_start = self.mark_height_offset, self.mark_width_offset
        h_end, w_end = h_start + self.mark_height, w_start + self.mark_width
        mask[h_start:h_end, w_start:w_end].copy_(self.mark[-1])
        return mask

    @staticmethod
    def scatter_mark(mark_unscattered: torch.Tensor,
                     mark_scattered_shape: list[int]) -> torch.Tensor:
        r"""Scatter the original mark tensor to a provided shape.

        If the shape are the same, it becomes a pixel shuffling process.

        Args:
            mark_unscattered (torch.Tensor): The unscattered mark tensor
                with shape ``(data_shape[0] + 1, mark_height, mark_width)``
            mark_scattered_shape (list[int]): The scattered mark shape
                ``(data_shape[0] + 1, mark_scattered_height, mark_scattered_width)``

        Returns:
            torch.Tensor: The scattered mark with shape :attr:`mark_scattered_shape`.
        """
        assert mark_scattered_shape[1] >= mark_unscattered.size(1), \
            f'mark_scattered_height={mark_scattered_shape[1]:d}  >=  mark_height={mark_unscattered.size(1):d}'
        assert mark_scattered_shape[2] >= mark_unscattered.size(2), \
            f'mark_scattered_width={mark_scattered_shape[2]:d}  >=  mark_width={mark_unscattered.size(2):d}'
        pixel_num = mark_unscattered[0].numel()
        mark = torch.zeros(mark_scattered_shape, device=env['device'])
        idx = torch.randperm(mark[0].numel())[:pixel_num]
        mark.flatten(1)[:, idx].copy_(mark_unscattered.flatten(1))
        return mark

    # ------------------------------ I/O --------------------------- #

    def load_mark(
        self,
        mark_img: str | Image.Image | np.ndarray | torch.Tensor,
        mark_background_color: None | str | torch.Tensor = 'auto',
        already_processed: bool = False
    ) -> torch.Tensor:
        r"""Load watermark tensor from image :attr:`mark_img`,
        scale by calling :any:`PIL.Image.Image.resize`
        and transform to ``(channel + 1, height, width)`` with alpha channel.

        Args:
            mark_img (PIL.Image.Image | str): Pillow image instance or file path.
            mark_background_color (str | torch.Tensor | None): Mark background color.
                If :class:`str`, choose from ``['auto', 'black', 'white']``;
                else, it shall be 1-dim tensor ranging in ``[0, 1]``.
                It's ignored when alpha channel in watermark image.
                Defaults to ``'auto'``.
            already_processed (bool):
                If ``True``, will just load :attr:`mark_img` as :attr:`self.mark`.
                Defaults to ``False``.

        Returns:
            torch.Tensor:
                Watermark tensor ranging in ``[0, 1]``
                with shape ``(channel + 1, height, width)`` with alpha channel.
        """
        if isinstance(mark_img, str):
            if mark_img.endswith('.npy'):
                mark_img = np.load(mark_img)
            else:
                if not os.path.isfile(mark_img) and \
                        not os.path.isfile(mark_img := os.path.join(dir_path, mark_img)):
                    raise FileNotFoundError(mark_img.removeprefix(dir_path))
                mark_img = F.convert_image_dtype(F.pil_to_tensor(Image.open(mark_img)))
        if isinstance(mark_img, np.ndarray):
            mark_img = torch.from_numpy(mark_img)
        mark: torch.Tensor = mark_img.to(device=env['device'])
        if not already_processed:
            mark = F.resize(mark, size=(self.mark_width, self.mark_height))
            alpha_mask = torch.ones_like(mark[0])
            if mark.size(0) == 4:
                mark = mark[:-1]
                alpha_mask = mark[-1]
            if self.data_shape[0] == 1 and mark.size(0) == 3:
                mark = F.rgb_to_grayscale(mark, num_output_channels=1)
            mark = torch.cat([mark, alpha_mask.unsqueeze(0)])

            if mark_background_color is not None:
                mark = update_mark_alpha_channel(mark, get_edge_color(mark, mark_background_color))
            if self.mark_random_init:
                mark[:-1] = torch.rand_like(mark[:-1])

            if self.mark_scattered:
                mark_scattered_shape = [mark.size(0), self.mark_scattered_height, self.mark_scattered_width]
                mark = self.scatter_mark(mark, mark_scattered_shape)
        self.mark_height, self.mark_width = mark.shape[-2:]
        self.mark = mark
        return mark


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    r"""
    | Add watermark arguments to argument parser.
    | For specific arguments implementation, see :meth:`Watermark.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
    """
    group = parser.add_argument_group('{yellow}mark{reset}'.format(**ansi))
    return Watermark.add_argument(group)


def create(mark_path: str = None, data_shape: list[int] = None,
           dataset_name: str = None, dataset: str | ImageSet = None,
           config: Config = config, **kwargs) -> Watermark:
    r"""
    | Create a watermark instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | For watermark implementation, see :class:`Watermark`.

    Args:
        mark_path (str):
            | Path to watermark image or npy file.
              There are some preset marks in the package.
            | Defaults to ``'square_white.png'``.
        data_shape (list[int]): The shape of image data ``[C, H, W]``.
        dataset_name (str): The dataset name.
        dataset (str): The alias of `dataset_name`.
        config (Config): The default parameter config.
        **kwargs: Keyword arguments
            passed to dataset init method.

    Returns:
        Watermark: Watermark instance.
    """
    if data_shape is None:
        assert isinstance(dataset, ImageSet), 'Please specify data_shape or dataset'
        data_shape = dataset.data_shape
    if dataset_name is None and dataset is not None:
        dataset_name = dataset.name
    result = config.get_config(dataset_name=dataset_name)[
        'mark'].update(kwargs).update(mark_path=mark_path)
    return Watermark(data_shape=data_shape, **result)
