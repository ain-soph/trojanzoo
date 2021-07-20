#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
import torchvision.models
from torchvision.models.vgg import model_urls as urls


class _VGG(_ImageModel):

    def __init__(self, name: str = 'vgg', **kwargs):
        if '_comp' in name:
            comp_dict = {'conv_dim': 512, 'fc_depth': 3, 'fc_dim': 512}
            if '_s' in name:
                comp_dict['fc_depth'] = 1
                comp_dict['fc_dim'] = 0
            for key, value in comp_dict.items():
                if key not in kwargs.keys():
                    kwargs[key] = value
        super().__init__(**kwargs)
        class_name = name.replace('_comp', '').replace('_s', '')
        ModelClass: type[torchvision.models.VGG] = getattr(torchvision.models, class_name)
        _model = ModelClass(num_classes=self.num_classes)
        self.features: nn.Sequential = _model.features
        if '_comp' in name:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = _model.avgpool   # nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = _model.classifier
        # nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # ))


class VGG(ImageModel):
    available_models = ['vgg', 'vgg_bn', 'vgg_comp', 'vgg_bn_comp',
                        'vgg11', 'vgg13', 'vgg16', 'vgg19',
                        'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                        'vgg11_comp', 'vgg13_comp', 'vgg16_comp', 'vgg19_comp',
                        'vgg11_bn_comp', 'vgg13_bn_comp', 'vgg16_bn_comp', 'vgg19_bn_comp'
                        'vgg11_comp_s', 'vgg13_comp_s', 'vgg16_comp_s', 'vgg19_comp_s',
                        'vgg11_bn_comp_s', 'vgg13_bn_comp_s', 'vgg16_bn_comp_s', 'vgg19_bn_comp_s']

    model_urls = urls

    def __init__(self, name: str = 'vgg', layer: int = 13,
                 model: type[_VGG] = _VGG, **kwargs):
        super().__init__(name=name, layer=layer, model=model, **kwargs)
