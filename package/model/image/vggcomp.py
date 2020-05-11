# -*- coding: utf-8 -*-
from .vgg import VGG
from ..image_cnn import _Image_CNN, Image_CNN

from package.imports.universal import *
from collections import OrderedDict

from torchvision.models.vgg import model_urls
import torchvision.models as models
from torch.utils import model_zoo


class _VGGcomp(_Image_CNN):
    """docstring for VGGcomp"""

    def __init__(self, layer=13, **kwargs):
        super(_VGGcomp, self).__init__(**kwargs)
        _model = models.__dict__[
            'vgg'+str(layer)](num_classes=self.num_classes)
        self.features = _model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class VGGcomp(VGG):
    """docstring for VGGcomp"""

    def __init__(self, name='vggcomp', layer=None, model_class=_VGGcomp, default_layer=13, **kwargs):
        super(VGGcomp, self).__init__(name=name, layer=layer, model_class=model_class, default_layer=default_layer,
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
