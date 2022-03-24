#!/usr/bin/env python3

from ...abstract import BackdoorDefense
from trojanvision.models import MagNet as MagNet_Model

import torch


class MagNet(BackdoorDefense):
    name: str = 'magnet'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.magnet = MagNet_Model(dataset=self.dataset, pretrained=True, folder_path=self.model.folder_path)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.validate_fn()

    def validate_fn(self, **kwargs) -> tuple[float, float]:
        clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                            get_data_fn=self.get_data, org=True, **kwargs)
        asr, _ = self.model._validate(print_prefix='Validate ASR',
                                      get_data_fn=self.get_data, keep_org=False, **kwargs)
        # self.model._validate(print_prefix='Validate Trigger Org',
        #                      get_data_fn=self.get_data, keep_org=False, poison_label=False, **kwargs)
        # print(f'Validate Confidence : {self.attack.validate_confidence():.3f}')
        return asr, clean_acc

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.attack.get_data(data=data, **kwargs)
        _input = self.magnet(_input)
        return _input, _label
