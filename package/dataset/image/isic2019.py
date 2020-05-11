# -*- coding: utf-8 -*-
from .isic import ISIC
from package.imports.universal import *


class ISIC2019(ISIC):
    """docstring for dataset"""

    def __init__(self, name='isic2019', num_classes=9, **kwargs):
        super(ISIC2019, self).__init__(
            name=name, num_classes=num_classes, **kwargs)
        self.url['train'] = 'https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip'
        self.org_folder_name['train'] = 'ISIC_2019_Training_Input'

        self.output_par(name='ISIC2019')