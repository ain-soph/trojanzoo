#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
import torchvision.models


cifar10_inverted_residual_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 1],    # Stride 2 -> 1 for CIFAR-10
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class _MobileNet(_ImageModel):
    def __init__(self, name: str = 'mobilenet_v2', **kwargs):
        try:
            sub_type: str = name[10:]
            assert sub_type in ['v2', 'v2_comp', 'v3_small', 'v3_large', 'v3_small_comp', 'v3_large_comp'], f'{name=}'
        except Exception:
            raise AssertionError(f'model name should be in {MobileNet.available_models}')
        super().__init__(**kwargs)
        if 'v2' in sub_type:
            inverted_residual_setting = cifar10_inverted_residual_setting if 'comp' in sub_type else None
            _model = torchvision.models.mobilenet.mobilenet_v2(num_classes=self.num_classes,
                                                               inverted_residual_setting=inverted_residual_setting)
        elif 'v3_large' in sub_type:
            _model = torchvision.models.mobilenet.mobilenet_v3_large(num_classes=self.num_classes)
        else:
            assert 'v3_small' in sub_type
            _model = torchvision.models.mobilenet.mobilenet_v3_small(num_classes=self.num_classes)
        if 'comp' in sub_type:
            conv1: nn.Conv2d = _model.features[0][0]
            _model.features[0][0] = nn.Conv2d(conv1.in_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                                              stride=1, padding=conv1.padding, dilation=conv1.dilation,
                                              groups=conv1.groups, bias=False)
        self.features = _model.features
        self.classifier = _model.classifier


class MobileNet(ImageModel):
    r"""MobileNets proposed by Andrew Howard and Liang-Chieh Chen from Google in CVPR 2018.

    :Available model names:

        .. code-block:: python3

            ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
             'mobilenet_v2_comp', 'mobilenet_v3_large_comp', 'mobilenet_v3_small_comp']

    See Also:
        * MobileNet v2:

          - torchvision: :any:`torchvision.models.mobilenet_v2`
          - paper: `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_
        * MobileNet v3:

          - torchvision: :any:`torchvision.models.mobilenet_v3_small`
          - paper: `Searching for MobileNetV3`_

    Note:
        ``_comp`` uses a small ``inverted_residual_setting``
        and set first conv layer ``stride=1``.

    .. _MobileNetV2\: Inverted Residuals and Linear Bottlenecks:
        https://arxiv.org/abs/1801.04381
    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """
    available_models = ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                        'mobilenet_v2_comp', 'mobilenet_v3_large_comp', 'mobilenet_v3_small_comp']

    model_urls = {
        'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
        'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
    }

    def __init__(self, name: str = 'mobilenet_v2', model: type[_MobileNet] = _MobileNet, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
