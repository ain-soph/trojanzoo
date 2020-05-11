# -*- coding: utf-8 -*-
from .isic import ISIC
from package.imports.universal import *


class ISIC2018(ISIC):
    """docstring for dataset"""

    def __init__(self, name='isic2018', num_classes=7, default_model='resnet101', **kwargs):
        super(ISIC2018, self).__init__(
            name=name, num_classes=num_classes, default_model=default_model, **kwargs)
        self.url['train'] = 'https://challenge.kitware.com/api/v1/item/5ac20fc456357d4ff856e139/download'
        self.org_folder_name['train'] = 'ISIC2018_Task3_Training_Input'

        self.output_par(name='ISIC2018')
