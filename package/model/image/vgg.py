# -*- coding: utf-8 -*-
from ..image_cnn import _Image_CNN, Image_CNN
from package.imports.universal import *

from collections import OrderedDict

from torchvision.models.vgg import model_urls
import torchvision.models as models
from torch.utils import model_zoo


class _VGG(_Image_CNN):
    """docstring for VGG"""

    # layer 13 or 16
    def __init__(self, layer=13, **kwargs):
        super(_VGG, self).__init__(**kwargs)
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


class VGG(Image_CNN):
    """docstring for VGG"""

    # layer 13 or 16
    def __init__(self, name='vgg', layer=None, model_class=_VGG, default_layer=13, **kwargs):
        name, layer = self.split_name(name=name, layer=layer, default_layer=default_layer)
        name = name+str(layer)
        self.layer = layer
        super(VGG, self).__init__(
            name=name, model_class=model_class, layer=layer, **kwargs)

    def load_official_weights(self, output=True):
        if output:
            print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls['vgg'+str(self.layer)])
        if self.num_classes == 1000:
            self._model.load_state_dict(_dict)
        else:
            new_dict = OrderedDict()
            for name, param in _dict.items():
                if 'classifier.6' not in name:
                    new_dict[name] = param
            self._model.load_state_dict(new_dict, strict=False)
