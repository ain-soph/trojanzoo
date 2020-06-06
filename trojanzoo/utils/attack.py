
# -*- coding: utf-8 -*-

from .tensor import to_tensor, byte2float

import os
import numpy as np
import torch
from typing import List, Union
from PIL.Image import Image

from trojanzoo import __file__ as root_file
root_dir = os.path.dirname(os.path.abspath(__file__))


# add mark to the Image with mask.
def add_mark(x: torch.Tensor, mark: torch.Tensor, _mask: torch.Tensor) -> torch.Tensor:
    result = x*(1-_mask)+mark*_mask
    return result


class Watermark:
    def __init__(self, data_shape: List[int], edge_color: Union[str, torch.Tensor] = 'auto',
                 mark_path: str = 'trojanzoo/data/mark/square_white.png', mark_alpha: float = 0.0,
                 height: int = 0, width: int = 0,
                 height_ratio: float = 1.0, width_ratio: float = 1.0,
                 height_offset: int = None, width_offset: int = None, **kwargs):

        if height == 0 and width == 0:
            height = int(height_ratio*data_shape[-2])
            width = int(width_ratio*data_shape[-1])
        # assert height != 0 and width != 0
        if height_offset is None:
            height_offset = data_shape[-2]-height
        if width_offset is None:
            width_offset = data_shape[-1]-width
        # assert height_offset is not None and height_offset is not None
        # --------------------------------------------------- #

        # WaterMark Image Parameters
        self.mark_alpha: float = mark_alpha
        self.data_shape: List[int] = data_shape
        self.mark_path: str = mark_path
        self.height: int = height
        self.width: int = width
        self.height_ratio: float = height_ratio
        self.width_ratio: float = width_ratio
        self.height_offset: int = height_offset
        self.width_offset: int = width_offset
        # --------------------------------------------------- #
        mark: torch.Tensor = self.load_img(mark_path, width, height)
        self.edge_color: torch.Tensor = self.get_edge_color(
            mark, data_shape, edge_color)

        self.mark, self.mask, self.alpha_mask = self.mask_mark(mark=mark)

    # add mark to the Image with mask.
    def add_mark(self, x: torch.Tensor, mark: torch.Tensor = None, _mask: torch.Tensor = None) -> torch.Tensor:
        if mark is None:
            mark = self.mark
        if _mask is None:
            _mask = self.mask*self.alpha_mask
        return add_mark(x, mark=mark, _mask=_mask)

    def load_file(self, mark_path: str):
        if mark_path[:9] == 'trojanzoo':
            mark_path = root_dir+mark_path[9:]
        _dict = np.load(mark_path)
        self.mark = to_tensor(_dict['mark'])
        self.mask = to_tensor(_dict['mask'])
        self.alpha_mask = to_tensor(_dict['alpha_mask'])

    @staticmethod
    def load_img(path: str, width: int, height: int) -> torch.Tensor:
        mark: Image = Image.open(path)
        mark = mark.resize((width, height), Image.ANTIALIAS)
        mark: torch.Tensor = byte2float(mark)
        return mark

    @staticmethod
    def get_edge_color(mark: torch.Tensor, data_shape: List[int],
                       edge_color: Union[str, torch.Tensor] = 'auto') -> torch.Tensor:

        assert data_shape[0] == mark.shape[1]

        t: torch.Tensor = torch.zeros(data_shape[0], dtype=torch.float)
        if isinstance(edge_color, str):
            if edge_color == 'black':
                pass
            elif edge_color == 'white':
                t += 1
            elif edge_color == 'auto':
                _list = (torch.as_tensor([mark[:, 0, :], mark[:, -1, :]]).view(data_shape[0], -1),
                         torch.as_tensor([mark[:, :, 0], mark[:, :, -1]]).view(data_shape[0], -1))
                _list = torch.cat(_list).view(data_shape[0], -1)
                t = _list.mode(dim=-1)[0]
            else:
                raise ValueError(edge_color)
        else:
            t = torch.as_tensor(edge_color)
            assert t.shape.item() == data_shape[0]
        return t

    # data_shape: channels, height, width
    # mark shape: channels, height, width
    # mask shape: channels, height, width
    # The mark shape may be smaller than the whole image. Fill the rest part as black, and return the mask and mark.
    def mask_mark(self, mark: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mask = torch.zeros(1, self.data_shape[-2], self.data_shape[-1],
                           dtype=torch.float)
        new_mark = -torch.ones(self.data_shape)
        for i in range(mark.shape[-2]):
            for j in range(mark.shape[-1]):
                if not mark[0, :, i, j].view(-1).equal():
                    mask[0, self.height_offset + i,
                         self.width_offset + j] = 1
                    new_mark[:, self.height_offset + i,
                             self.width_offset + j] = mark[:, i, j]
        new_mark = to_tensor(new_mark.unsqueeze(0).detach())
        mask = to_tensor(mask.repeat(1, self.data_shape[0], 1, 1).detach())
        alpha_mask = to_tensor((mask*(1-self.mark_alpha)).detach())
        return new_mark, mask, alpha_mask
