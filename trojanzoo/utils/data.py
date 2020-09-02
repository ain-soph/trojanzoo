# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

import torch


class MyDataset(Dataset):
    def __init__(self, data: torch.FloatTensor, targets: torch.LongTensor):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
