# -*- coding: utf-8 -*-

from .attack import Attack
from trojanzoo.utils import to_tensor, read_img_as_tensor, byte2float, repeat_to_batch
from trojanzoo.utils.attack import add_mark

import os
from PIL.Image import Image
import random
from typing import Union, List

import numpy as np
import torch

from trojanzoo import __file__ as root_file
root_dir = os.path.dirname(os.path.abspath(__file__))


class Backdoor_Attack(Attack):

    name = 'backdoor'

    def __init__(self, watermark: Watermark = None, target_class: int = 0, alpha: float = 0.0, percent: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.watermark: Watermark = watermark
        self.target_class: int = target_class
        self.alpha: float = alpha
        self.percent: float = percent
        self.filename: str = self.get_filename()

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, iteration: int = None, **kwargs):
        if iteration is None:
            iteration = self.iteration
        self.model._train(epoch=iteration, optimizer=optimizer, lr_scheduler=lr_scheduler,
                          get_data=self.get_data, validate_func=self.validate_func, **kwargs)

    def add_mark(self, x, **kwargs):
        return self.watermark.add_mark(x, **kwargs)

    def get_filename(self, alpha: float = None, target_class: int = None, iteration: int = None):
        if alpha is None:
            alpha = self.alpha
        if target_class is None:
            target_class = self.target_class
        if iteration is None:
            iteration = self.iteration
        _file = '{mark}_tar{target:d}_alpha{alpha:.2f}_mark({height:d},{width:d})_iter{iteration:d}_percent{percent:.2f}'.format(
            mark=os.path.split(self.path)[1][:-4], target=target_class,
            alpha=alpha, iteration=iteration, percent=self.percent,
            height=self.watermark.height, width=self.watermark.width)
        return _file

    def get_data(self, data: (torch.Tensor, torch.LongTensor), keep_org: bool = True) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)
        if not keep_org or random.uniform(0, 1) < self.percent:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input)
            _label = self.target_class*torch.ones_like(org_label)
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_func(self, **kwargs) -> (float, float, float):
        self.model._validate(print_prefix='Validate Clean',
                             get_data=None, **kwargs)
        self.model._validate(print_prefix='Validate Watermark',
                             keep_org=False, **kwargs)
        return 0.0, 0.0, 0.0


class Watermark:
    def __init__(self, data_shape: List[int], edge_color: Union[str, torch.Tensor] = 'auto',
                 path: str = root_dir+'/data/mark/square_white.png',
                 height: int = 0, width: int = 0,
                 height_ratio: float = 1.0, width_ratio: float = 1.0,
                 height_offset: int = None, width_offset: int = None):

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
        self.data_shape: List[int] = data_shape
        self.path: str = path
        self.height: int = height
        self.width: int = width
        self.height_ratio: float = height_ratio
        self.width_ratio: float = width_ratio
        self.height_offset: int = height_offset
        self.width_offset: int = width_offset
        # --------------------------------------------------- #
        mark: torch.Tensor = self.load_img(path, width, height)
        self.edge_color: torch.Tensor = self.get_edge_color(
            mark, data_shape, edge_color)

        self.mark, self.mask, self.alpha_mask = self.mask_mark(mark=mark)

    # add mark to the Image with mask.
    def add_mark(self, x, mark: torch.Tensor = None, _mask: torch.Tensor = None):
        if mark is None:
            mark = self.mark
        if _mask is None:
            _mask = self.mask*self.alpha_mask
        return add_mark(x, mark=mark, _mask=_mask)

    def load_file(self, path: str):
        _dict = np.load(path)
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
        alpha_mask = to_tensor((mask*(1-self.alpha)).detach())
        return new_mark, mask, alpha_mask
