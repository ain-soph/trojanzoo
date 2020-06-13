# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Process
from trojanzoo.dataset import Dataset, ImageSet
from trojanzoo.model import Model, ImageModel

import os
import torch

from trojanzoo.utils import Config
env = Config.env


class Attack(Process):

    name: str = 'attack'

    def __init__(self, dataset: ImageSet = None, model: ImageModel = None, folder_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['attack'] = ['folder_path']
        self.dataset = dataset
        self.model = model

        # ----------------------------------------------------------------------------- #
        if folder_path is None:
            folder_path = env['result_dir'] + self.name + '/'
            if dataset and isinstance(dataset, Dataset):
                folder_path += dataset.name + '/'
            if model and isinstance(model, Model):
                folder_path += model.name + '/'
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def attack(self, **kwargs):
        pass

    # ----------------------Utility----------------------------------- #
    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.LongTensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)
