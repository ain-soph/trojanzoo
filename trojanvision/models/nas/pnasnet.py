#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs.pnasnet import PNASNetA, PNASNetB

import torch.nn as nn
from collections import OrderedDict


class _PNASNet(_ImageModel):

    def __init__(self, name: str = None, **kwargs):
        super().__init__(**kwargs)
        ModelClass = PNASNetA if '_a' in name else PNASNetB
        _model = ModelClass(num_classes=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),
            ('relu', nn.ReLU(inplace=True)),
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4),
            ('layer5', _model.layer5)
        ]))
        # self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.linear)
        ]))


class PNASNet(ImageModel):
    r"""PNASNet proposed by Chenxi Liu from Johns Hopkins University in ECCV 2018.

    Note:
        The implementation is imported from a third-party github repo. The correctness can't be guaranteed.
        It might be better to reimplement according to tensorflow codes:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py

    :Available model names:

        .. code-block:: python3

            ['pnasnet', 'pnasnet_a', 'pnasnet_b']

    See Also:
        * paper: `Progressive Neural Architecture Search`_
        * code: https://github.com/kuangliu/pytorch-cifar/blob/master/models/pnasnet.py

    .. _Progressive Neural Architecture Search:
        https://arxiv.org/abs/1712.00559
    """
    available_models = ['pnasnet', 'pnasnet_a', 'pnasnet_b']

    def __init__(self, name: str = 'pnasnet', layer: str = '_b',
                 model: type[_PNASNet] = _PNASNet, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)

    @classmethod
    def get_name(cls, name: str, layer: str = '') -> str:
        layer = layer if name == 'pnasnet' else ''
        return super().get_name(name, layer=layer)
