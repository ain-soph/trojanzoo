#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs import dla

import torch.nn as nn

import torch
from collections import OrderedDict
from collections.abc import Callable

urls = {
    'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth',
    'dla46_c': 'http://dl.yf.io/dla/models/imagenet/dla46_c-2bfd52c3.pth',
    'dla46x_c': 'http://dl.yf.io/dla/models/imagenet/dla46x_c-d761bae7.pth',
    'dla60x_c': 'http://dl.yf.io/dla/models/imagenet/dla60x_c-b870c45c.pth',
    'dla60': 'http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth',
    'dla60x': 'http://dl.yf.io/dla/models/imagenet/dla60x-d15cacda.pth',
    'dla102': 'http://dl.yf.io/dla/models/imagenet/dla102-d94d9790.pth',
    'dla102x': 'http://dl.yf.io/dla/models/imagenet/dla102x-ad62be81.pth',
    'dla102x2': 'http://dl.yf.io/dla/models/imagenet/dla102x2-262837b6.pth',
    'dla169': 'http://dl.yf.io/dla/models/imagenet/dla169-0914e092.pth',
    # 'dla60_res2net':
    #     'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net_dla60_4s-d88db7f9.pth',
    # 'dla60_res2next':
    #     'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2next_dla60_4s-d327927b.pth',
}


class _DLA(_ImageModel):

    def __init__(self, name: str = 'dla', **kwargs):
        super().__init__(**kwargs)
        ModelClass: Callable[..., dla.DLA] = getattr(dla, name.replace('_comp', ''))
        if 'comp' in name:
            _model = ModelClass(num_classes=self.num_classes, strides=[1, 2, 2, 2])
            conv1: nn.Conv2d = _model.features[0][0]
            _model.features[0][0] = dla.conv3x3(conv1.in_channels, conv1.out_channels)  # stem.conv kernel_size: 7 -> 3
            conv3: nn.Conv2d = _model.features[2][0]
            _model.features[2][0] = dla.conv3x3(conv3.in_channels, conv3.out_channels)  # layer2.conv stride: 2 -> 1
        else:
            _model = ModelClass(num_classes=self.num_classes)
        self.features = _model.features
        self.classifier = _model.classifier


class DLA(ImageModel):
    available_models = ['dla', 'dla_comp',
                        'dla34', 'dla46_c', 'dla46x_c', 'dla60x_c', 'dla60', 'dla60x',
                        'dla102', 'dla102x', 'dla102x2', 'dla169',
                        'dla34_comp', 'dla46_c_comp', 'dla46x_c_comp', 'dla60x_c_comp', 'dla60_comp', 'dla60x_comp',
                        'dla102_comp', 'dla102x_comp', 'dla102x2_comp', 'dla169_comp']
    model_urls = urls

    def __init__(self, name: str = 'dla', layer: int = 34, model: type[_DLA] = _DLA, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        old_dict = super().get_official_weights(**kwargs)
        new_dict: OrderedDict[str, torch.Tensor] = self.state_dict()
        old_keys = list(old_dict.keys())
        new_keys = list(new_dict.keys())
        new2old: dict[str, str] = {}
        i = 0
        j = 0
        while(i < len(new_keys) and j < len(old_keys)):
            if 'num_batches_tracked' in new_keys[i]:
                i += 1
                continue
            if 'project' not in new_keys[i] and 'project' in old_keys[j]:
                j += 1
                continue
            # print(f'{new_keys[i]:60} {old_keys[j]}')
            new2old[new_keys[i]] = old_keys[j]
            i += 1
            j += 1
        for i, key in enumerate(new_keys):
            if 'num_batches_tracked' in key:
                new_dict[key] = torch.tensor(0)
            elif 'fc.weight' in key:
                weight = old_dict[new2old[key]]
                new_dict[key] = weight.flatten(1)
            else:
                new_dict[key] = old_dict[new2old[key]]
        return new_dict
