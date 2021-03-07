#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel
from trojanvision.datasets import ImageSet
from trojanvision.utils.darts import FeatureExtractor, Genotype
from trojanvision.utils.darts import DARTS as darts_genotype

import torch
from torchvision.datasets.utils import download_file_from_google_drive
import os
from collections import OrderedDict

url = {
    'cifar10': '1Y13i4zKGKgjtWBdC0HWLavjO7wvEiGOc',
    'ptb': '1Mt_o6fZOlG-VDF3Q5ModgnAJ9W6f_av2',
    'imagenet': '1AKr6Y_PoYj7j0Upggyzc26W0RVdg4CVX'
}


class _DARTS(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features: FeatureExtractor
        self.classifier = self.define_classifier(conv_dim=self.features.feats_dim,
                                                 num_classes=self.num_classes, fc_depth=1)

    @staticmethod
    def define_features(genotype: Genotype = darts_genotype,
                        C: int = 36, layer: int = 20,
                        dropout_p: float = 0.2, **kwargs) -> FeatureExtractor:
        return FeatureExtractor(genotype, C, layer, dropout_p)

    def get_fm(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.normalize(x))[0]


class DARTS(ImageModel):
    def __init__(self, name: str = 'darts', layer: int = 20,
                 model_class: type[_DARTS] = _DARTS, **kwargs):
        # TODO: ImageNet parameter settings
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)

    def get_official_weights(self, dataset='cifar10', auxiliary: bool = False,
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        file_name = f'darts_{dataset}.pt'
        download_file_from_google_drive(file_id=url[dataset], root=self.folder_path, filename=file_name)
        print('get official model weights from Google Drive: ', url[dataset])
        _dict: OrderedDict[str, torch.Tensor] = torch.load(os.path.join(self.folder_path, file_name),
                                                           map_location='cpu')
        if 'state_dict' in _dict.keys():
            _dict = _dict['state_dict']

        new_dict: OrderedDict[str, torch.Tensor] = self.state_dict()
        old_keys = list(_dict.keys())
        new_keys = list(new_dict.keys())
        new2old: dict[str, str] = {}
        i = 0
        j = 0
        while(i < len(new_keys) and j < len(old_keys)):
            if 'num_batches_tracked' in new_keys[i]:
                i += 1
                continue
            if not auxiliary and 'auxiliary_head' in old_keys[j]:
                j += 1
                continue
            new2old[new_keys[i]] = old_keys[j]
            i += 1
            j += 1
        for i, key in enumerate(new_keys):
            if 'num_batches_tracked' in key:
                new_dict[key] = torch.tensor(0)
            else:
                new_dict[key] = _dict[new2old[key]]
        return new_dict
