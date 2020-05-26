# -*- coding: utf-8 -*-

from .dataset import Dataset
from trojanzoo.utils import to_tensor

import torch

from trojanzoo.config import Config
config = Config.config


class ImageSet(Dataset):
    """docstring for dataset"""

    def __init__(self, name='imageset', n_channel=3, n_dim=(0, 0), norm_par: dict = None, default_model='resnetcomp18', **kwargs):
        self.norm_par = norm_par
        self.n_channel = n_channel
        self.n_dim = n_dim
        super(ImageSet, self).__init__(
            name=name, data_type='image', default_model=default_model, **kwargs)

    def get_dataloader(self, mode, full=False, batch_size: int = None, shuffle: bool = None, num_workers: int = None, **kwargs):
        if batch_size is None:
            if mode == 'test':
                batch_size = 1
            else:
                batch_size = self.batch_size
        if shuffle is None:
            shuffle = True if mode == 'train' else False
        if num_workers is None:
            num_workers = self.num_workers

        dataset = self.get_dataset(mode, full=full, **kwargs)
        torch.manual_seed(config['general']['seed']['torch'])
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @staticmethod
    def get_data(data, **kwargs):
        return to_tensor(data[0]), to_tensor(data[1], dtype='long')
