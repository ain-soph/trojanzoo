
# -*- coding: utf-8 -*-

from trojanzoo import __file__ as root_file
from .tensor import to_tensor, to_numpy, byte2float, gray_img, save_tensor_as_img
from .output import prints, Indent_Redirect

import os
import random
import numpy as np
import torch
from PIL import Image
from collections import OrderedDict
from typing import List, Union

from trojanzoo.utils import Config
env = Config.env

root_dir = os.path.dirname(os.path.abspath(root_file))

redirect = Indent_Redirect(buffer=True, indent=0)


class Watermark:
    name: str = 'mark'

    def __init__(self, data_shape: List[int], edge_color: Union[str, torch.Tensor] = 'auto',
                 mark_path: str = 'trojanzoo/data/mark/square_white.png', mark_alpha: float = 0.0,
                 height: int = None, width: int = None,
                 height_ratio: float = None, width_ratio: float = None, mark_ratio: float = None,
                 height_offset: int = 0, width_offset: int = 0,
                 random_pos=False, random_init=False, **kwargs):

        self.param_list: Dict[str, List[str]] = OrderedDict()
        self.param_list['mark'] = ['mark_path', 'data_shape', 'edge_color',
                                   'mark_alpha', 'height', 'width',
                                   'random_pos', 'random_init']
        if height is None and width is None:
            if height_ratio is None and width_ratio is None:
                assert mark_ratio is not None
                self.param_list['mark'].append('mark_ratio')
                self.mark_ratio = mark_ratio
                height_ratio = mark_ratio
                width_ratio = mark_ratio
            height = int(height_ratio * data_shape[-2])
            width = int(width_ratio * data_shape[-1])
        assert height > 0 and width > 0
        # --------------------------------------------------- #

        # WaterMark Image Parameters
        self.mark_alpha: float = mark_alpha
        self.data_shape: List[int] = data_shape
        self.mark_path: str = mark_path
        self.height: int = height
        self.width: int = width
        self.random_pos = random_pos
        # --------------------------------------------------- #
        org_mark_img: Image.Image = self.load_img(mark_path, height, width, data_shape[0])
        self.org_mark: torch.Tensor = byte2float(org_mark_img)
        self.edge_color: torch.Tensor = self.get_edge_color(
            self.org_mark, data_shape, edge_color)
        self.org_mask, self.org_alpha_mask = self.org_mask_mark(self.org_mark, self.edge_color, self.mark_alpha)
        self.random_init = random_init
        if random_init:
            self.org_mark = self.random_init_mark(self.org_mark, self.org_mask)

        if not random_pos:
            self.param_list['mark'].extend(['height_offset', 'width_offset'])
            self.height_offset: int = height_offset
            self.width_offset: int = width_offset
            self.mark, self.mask, self.alpha_mask = self.mask_mark(
                height_offset=self.height_offset, width_offset=self.width_offset)

    # add mark to the Image with mask.
    def add_mark(self, _input: torch.Tensor, random_pos=None, **kwargs) -> torch.Tensor:
        if random_pos is None:
            random_pos = self.random_pos
        if random_pos:
            batch_size = _input.size(0)
            # height_offset = torch.randint(high=self.data_shape[-2] - self.height, size=[batch_size])
            # width_offset = torch.randint(high=self.data_shape[-1] - self.width, size=[batch_size])
            height_offset = random.randint(0, self.data_shape[-2] - self.height)
            width_offset = random.randint(0, self.data_shape[-1] - self.width)
            mark, mask, alpha_mask = self.mask_mark(height_offset=height_offset, width_offset=width_offset)
        else:
            mark, mask, alpha_mask = self.mark, self.mask, self.alpha_mask
        _mask = mask * alpha_mask
        mark, _mask = mark.to(_input.device), _mask.to(_input.device)
        return _input + _mask * (mark - _input)

    @staticmethod
    def get_edge_color(mark: torch.Tensor, data_shape: List[int],
                       edge_color: Union[str, torch.Tensor] = 'auto') -> torch.Tensor:

        assert data_shape[0] == mark.shape[0]
        t: torch.Tensor = torch.zeros(data_shape[0], dtype=torch.float)
        if isinstance(edge_color, str):
            if edge_color == 'black':
                pass
            elif edge_color == 'white':
                t += 1
            elif edge_color == 'auto':
                mark = mark.transpose(0, -1)
                if mark.flatten(start_dim=1).std(dim=1).max() < 1e-3:
                    t = -torch.ones_like(mark[0, 0])
                else:
                    _list = [mark[0, :, :], mark[-1, :, :],
                             mark[:, 0, :], mark[:, -1, :]]
                    _list = torch.cat(_list)
                    t = _list.mode(dim=0)[0]
            else:
                raise ValueError(edge_color)
        else:
            t = torch.as_tensor(edge_color)
            assert t.shape.item() == data_shape[0]
        return t

    @staticmethod
    def org_mask_mark(org_mark: torch.Tensor, edge_color: torch.Tensor, mark_alpha: float) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        height, width = org_mark.shape[-2:]
        mark = torch.zeros_like(org_mark, dtype=torch.float)
        mask = torch.zeros([height, width], dtype=torch.bool)
        for i in range(height):
            for j in range(width):
                if not org_mark[:, i, j].equal(edge_color):
                    mark[:, i, j] = org_mark[:, i, j]
                    mask[i, j] = 1
        alpha_mask = mask * (1 - mark_alpha)
        return mask, alpha_mask

    def mask_mark(self, height_offset: int, width_offset: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mark = -torch.ones(self.data_shape, dtype=torch.float)
        mask = torch.zeros(self.data_shape[-2:], dtype=torch.bool)
        alpha_mask = torch.zeros_like(mask, dtype=torch.float)

        start_h = height_offset
        start_w = width_offset
        end_h = height_offset + self.height
        end_w = width_offset + self.width

        mark[:, start_h:end_h, start_w:end_w] = self.org_mark
        mask[start_h:end_h, start_w:end_w] = self.org_mask
        alpha_mask[start_h:end_h, start_w:end_w] = self.org_alpha_mask
        if env['num_gpus']:
            mark = mark.to(env['device'])
            mask = mask.to(env['device'])
            alpha_mask = alpha_mask.to(env['device'])
        return mark, mask, alpha_mask

    """
    # each image in the batch has a unique random location.
    def mask_mark_batch(self, height_offset: torch.Tensor, width_offset: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        assert len(height_offset) == len(width_offset)
        shape = [len(height_offset)].extend(self.data_shape)
        mark = -torch.ones(shape, dtype=int)
        shape[1] = 1
        mask = torch.zeros(shape, dtype=torch.float)
        alpha_mask = torch.zeros_like(mask)

        start_h = height_offset
        start_w = width_offset
        end_h = height_offset + self.height
        end_w = width_offset + self.width

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
    def random_init_mark(mark, mask):
        init_mark = torch.rand_like(mark)
        ones = -torch.ones_like(mark)
        init_mark = torch.where(mask, init_mark, ones)
        return init_mark

    # ------------------------------ I/O --------------------------- #

    @staticmethod
    def load_img(img_path: str, height: int, width: int, channel: int = 3) -> Image.Image:
        if img_path[:9] == 'trojanzoo':
            img_path = root_dir + img_path[9:]
        mark: Image.Image = Image.open(img_path)
        mark = mark.resize((width, height), Image.ANTIALIAS)

        if channel == 1:
            mark = gray_img(mark, num_output_channels=1)
        elif channel == 3 and mark.mode in ['1', 'L']:
            mark = gray_img(mark, num_output_channels=3)
        return mark

    def save_img(self, img_path: str):
        if img_path[:9] == 'trojanzoo':
            img_path = root_dir + img_path[9:]
        img = self.org_mark * self.org_mask if self.random_pos else self.mark * self.mask
        save_tensor_as_img(img_path, img)

    def load_npz(self, npz_path: str):
        if npz_path[:9] == 'trojanzoo':
            npz_path = root_dir + npz_path[9:]
        _dict = np.load(npz_path)
        self.org_mark = torch.as_tensor(_dict['org_mark'])
        self.org_mask = torch.as_tensor(_dict['org_mask'])
        self.org_alpha_mask = torch.as_tensor(_dict['org_alpha_mask'])
        if not self.random_pos:
            self.mark = to_tensor(_dict['mark'])
            self.mask = to_tensor(_dict['mask'])
            self.alpha_mask = to_tensor(_dict['alpha_mask'])

    def save_npz(self, npz_path: str):
        # if npz_path[:9] == 'trojanzoo':
        #     npz_path = root_dir + npz_path[9:]
        _dict = {'org_mark': to_numpy(self.org_mark),
                 'org_mask': to_numpy(self.org_mask),
                 'org_alpha_mask': to_numpy(self.org_alpha_mask)}
        if not self.random_pos:
            _dict.update({
                'mark': to_numpy(self.mark),
                'mask': to_numpy(self.mask),
                'alpha_mask': to_numpy(self.alpha_mask)
            })
        np.savez(npz_path, **_dict)

    # ------------------------------Verbose Information--------------------------- #
    def summary(self, indent: int = 0):
        prints(f'{self.name:<10s} Parameters: ', indent=indent)
        d = self.__dict__
        for key, value in self.param_list.items():
            prints(key, indent=indent + 10)
            prints({v: getattr(self, v) for v in value}, indent=indent + 10)
            prints('-' * 20, indent=indent + 10)

    def __str__(self) -> str:
        sys.stdout = redirect
        self.summary()
        _str = redirect.buffer
        redirect.reset()
        return _str
