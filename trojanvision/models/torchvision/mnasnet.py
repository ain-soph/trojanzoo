#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torchvision.models
from torchvision.models.mnasnet import MNASNet0_5_Weights, MNASNet0_75_Weights, MNASNet1_0_Weights, MNASNet1_3_Weights
import re

import torch
from collections import OrderedDict


class _MNASNet(_ImageModel):

    def __init__(self, mnas_alpha: float, **kwargs):
        super().__init__(**kwargs)
        _model = torchvision.models.MNASNet(mnas_alpha, num_classes=self.num_classes)
        self.features = _model.layers
        self.classifier = _model.classifier
        # conv: nn.Conv2d = self.features[0]
        # self.features[0] = nn.Conv2d(3, conv.out_channels, 3, padding=1, stride=1, bias=False)


class MNASNet(ImageModel):
    r"""MNASNet proposed by Mingxing Tan from Google in CVPR 2019.

    :Available model names:

        .. code-block:: python3

            {'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'}

    See Also:
        * torchvision: :any:`torchvision.models.mnasnet0_5`
        * paper: `MnasNet\: Platform-Aware Neural Architecture Search for Mobile`_

    .. _MnasNet\: Platform-Aware Neural Architecture Search for Mobile:
        https://arxiv.org/abs/1807.11626
    """
    available_models = {'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'}
    weights = {
        'mnasnet0_5': MNASNet0_5_Weights,
        'mnasnet0_75': MNASNet0_75_Weights,
        'mnasnet1_0': MNASNet1_0_Weights,
        'mnasnet1_3': MNASNet1_3_Weights,
    }

    def __init__(self, name: str = 'mnasnet', mnas_alpha: float = 1.0,
                 model: type[_MNASNet] = _MNASNet, **kwargs):
        name, self.mnas_alpha = self.parse_name(name, mnas_alpha)
        super().__init__(name=name, mnas_alpha=self.mnas_alpha, model=model, **kwargs)

    @staticmethod
    def parse_name(name: str, mnas_alpha: float = 1.0) -> tuple[str, float]:
        name_list: list[str] = re.findall(r'[a-zA-Z]+|[\d_.]+', name)
        name = name_list[0]
        if len(name_list) > 1:
            assert len(name_list) == 2
            mnas_alpha = float(name_list[1].replace('_', '.'))
        mnas_alpha_str = f'{mnas_alpha:.2f}'.removesuffix('0')
        return f'{name}{mnas_alpha_str}'.replace('.', '_'), mnas_alpha

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        weights = getattr(self.weights, self.parse_name('mnasnet', self.mnas_alpha)[0])
        _dict = super().get_official_weights(weights=weights)
        new_dict = OrderedDict()
        for key, value in _dict.items():
            if key.startswith('layers.'):
                key = 'features.' + key[7:]
            new_dict[key] = value
        return new_dict
