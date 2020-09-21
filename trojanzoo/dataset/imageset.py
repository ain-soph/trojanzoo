# -*- coding: utf-8 -*-

from .dataset import Dataset
from trojanzoo.utils import to_tensor

import torch
import torchvision.transforms as transforms
from typing import Union, List, Tuple, Dict

from trojanzoo.utils.config import Config
env = Config.env


class ImageSet(Dataset):

    name: str = 'imageset'
    data_type: str = 'image'
    n_channel: int = 3
    n_dim: Tuple[int] = (0, 0)

    def __init__(self, norm_par: Dict[str, List[float]] = None,
                 default_model: str = 'resnetcomp18', **kwargs):
        super().__init__(default_model=default_model, **kwargs)
        self.norm_par: Dict[str, List[float]] = norm_par
        self.param_list['imageset'] = ['n_channel', 'n_dim', 'norm_par']

    @classmethod
    def get_transform(cls, **kwargs):
        return transforms.ToTensor()

    def get_dataloader(self, mode: str, dataset: Dataset = None, batch_size: int = None, shuffle: bool = None,
                       num_workers: int = None, pin_memory=True, drop_last=False, **kwargs):
        if batch_size is None:
            batch_size = 1 if mode == 'test' else self.batch_size
        if shuffle is None:
            shuffle = True if mode == 'train' else False
        if num_workers is None:
            num_workers = self.num_workers if mode == 'train' else 0
        if dataset is None:
            dataset = self.get_dataset(mode, **kwargs)
        if env['num_gpus'] == 0:
            pin_memory = False
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    @staticmethod
    def get_data(data: Tuple[torch.Tensor], **kwargs) -> (torch.Tensor, torch.LongTensor):
        return to_tensor(data[0]), to_tensor(data[1], dtype='long')
