#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanvision.environ import env
from trojanzoo.utils import to_numpy

import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm


class STRIP(BackdoorDefense):
    name: str = 'strip'

    def __init__(self, alpha: float = 0.5, N: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.N: int = N
        self.loader = self.dataset.get_dataloader(mode='train', drop_last=True)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        clean_entropy = []
        poison_entropy = []
        loader = self.dataset.loader['valid']
        if env['tqdm']:
            loader = tqdm(loader)
        for i, data in enumerate(loader):
            _input, _label = self.model.get_data(data)
            poison_input = self.attack.add_mark(_input)
            clean_entropy.append(self.check(_input, _label))
            poison_entropy.append(self.check(poison_input, _label))
        clean_entropy = torch.cat(clean_entropy).flatten().sort()[0]
        poison_entropy = torch.cat(poison_entropy).flatten().sort()[0]
        _dict = {'clean': to_numpy(clean_entropy), 'poison': to_numpy(poison_entropy)}
        result_file = f'{self.folder_path}{self.get_filename()}.npy'
        np.save(result_file, _dict)
        print('File Saved at : ', result_file)
        print('Entropy Clean  Median: ', float(clean_entropy.median()))
        print('Entropy Poison Median: ', float(poison_entropy.median()))

        threshold_low = float(clean_entropy[int(0.05 * len(clean_entropy))])
        threshold_high = float(clean_entropy[int(0.95 * len(clean_entropy))])
        y_true = torch.cat((torch.zeros_like(clean_entropy), torch.ones_like(poison_entropy)))
        entropy = torch.cat((clean_entropy, poison_entropy))
        y_pred = torch.where(((entropy < threshold_low).int() + (entropy > threshold_high).int()).bool(),
                             torch.ones_like(entropy), torch.zeros_like(entropy))
        print(f'Threshold: ({threshold_low:5.3f}, {threshold_high:5.3f})')
        print("f1_score:", metrics.f1_score(y_true, y_pred))
        print("precision_score:", metrics.precision_score(y_true, y_pred))
        print("recall_score:", metrics.recall_score(y_true, y_pred))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))

    def check(self, _input: torch.Tensor, _label: torch.Tensor) -> torch.Tensor:
        _list = []
        for i, data in enumerate(self.loader):
            if i >= self.N:
                break
            X, Y = self.model.get_data(data)
            _test = self.superimpose(_input, X)
            entropy = self.entropy(_test).cpu()
            _list.append(entropy)
            # _class = self.model.get_class(_test)
        return torch.stack(_list).mean(0)

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        _input2 = _input2[:_input1.shape[0]]

        result = alpha * (_input1 - _input2) + _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        p = self.model.get_prob(_input)
        return (-p * p.log()).sum(1)
