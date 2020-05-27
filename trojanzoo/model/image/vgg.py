# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

from collections import OrderedDict

import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
import torchvision.models as models


class _VGG(_ImageModel):

    # layer 13 or 16
    def __init__(self, layer=13, **kwargs):
        super().__init__(**kwargs)
        _model = models.__dict__[
            'vgg'+str(layer)](num_classes=self.num_classes)
        self.features = _model.features
        self.avgpool = _model.avgpool   # nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = _model.classifier

        # nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )


class VGG(ImageModel):

    # layer 13 or 16
    def __init__(self, name='vgg', layer=None, model_class=_VGG, default_layer=13, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['vgg'+str(self.layer)]
        _dict = model_zoo.load_url(url)
        if self.num_classes == 1000:
            self._model.load_state_dict(_dict)
        else:
            new_dict = OrderedDict()
            for name, param in _dict.items():
                if 'classifier.6' not in name:
                    new_dict[name] = param
            self._model.load_state_dict(new_dict, strict=False)
        if verbose:
            print(
                'Model {name} loaded From Official Website: '.format(self.name), url)


class _VGGcomp(_VGG):

    def __init__(self, **kwargs):
        super(_VGGcomp, self).__init__(**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class VGGcomp(VGG):

    def __init__(self, name='vggcomp', model_class=_VGGcomp, **kwargs):
        super(VGGcomp, self).__init__(name=name, model_class=model_class,
                                      conv_dim=512, fc_depth=3, fc_dim=512, **kwargs)

    def load_official_weights(self, output=True):
        if output:
            print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls['vgg'+str(self.layer)])
        new_dict = OrderedDict()
        for name, param in _dict.items():
            if 'classifier' not in name:
                new_dict[name] = param
        self._model.load_state_dict(new_dict, strict=False)

    def load_official_weights(self, verbose=True):
        url = model_urls['vgg'+str(self.layer)]
        _dict = model_zoo.load_url(url)
        new_dict = OrderedDict()
        for name, param in _dict.items():
            if 'classifier' not in name:
                new_dict[name] = param
        self._model.load_state_dict(new_dict, strict=False)
        if verbose:
            print(
                'Model {name} loaded From Official Website: '.format(self.name), url)
