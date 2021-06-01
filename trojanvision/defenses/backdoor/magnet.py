#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanvision.models import MagNet as MagNet_Model

import torch


class MagNet(BackdoorDefense):
    name: str = 'magnet'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.magnet = MagNet_Model(dataset=self.dataset, pretrain=True, folder_path=self.model.folder_path)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.validate_fn()

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label=True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        if org:
            _input, _label = self.model.get_data(data)
        else:
            _input, _label = self.attack.get_data(data=data, keep_org=keep_org, poison_label=poison_label, **kwargs)
        _input = self.magnet(_input)
        return _input, _label

    def validate_fn(self, **kwargs) -> tuple[float, float]:
        _, clean_acc = self.model._validate(print_prefix='Validate Clean',
                                            get_data_fn=self.get_data, org=True, **kwargs)
        _, target_acc = self.model._validate(print_prefix='Validate Trigger Tgt',
                                             get_data_fn=self.get_data, keep_org=False, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org',
                             get_data_fn=self.get_data, keep_org=False, poison_label=False, **kwargs)
        print(f'Validate Confidence : {self.attack.validate_confidence():.3f}')
        return clean_acc, target_acc
