#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel
from trojanvision.utils.darts import FeatureExtractor, AuxiliaryHead, Genotype
from trojanvision.utils.darts import DARTS as DARTS_genotype
from trojanvision.utils.darts import ROBUST_DARTS

import torch
from torchvision.datasets.utils import download_file_from_google_drive
import os
from collections import OrderedDict

import argparse  # TODO: python 3.10

url = {
    'cifar10': '1Y13i4zKGKgjtWBdC0HWLavjO7wvEiGOc',
    'ptb': '1Mt_o6fZOlG-VDF3Q5ModgnAJ9W6f_av2',
    'imagenet': '1AKr6Y_PoYj7j0Upggyzc26W0RVdg4CVX'
}


class _DARTS(_ImageModel):
    def __init__(self, auxiliary: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.features: FeatureExtractor
        self.classifier = self.define_classifier(conv_dim=self.features.feats_dim,
                                                 num_classes=self.num_classes, fc_depth=1)
        self.auxiliary_head: AuxiliaryHead = None
        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C=self.features.feats_dim, num_classes=self.num_classes)

    @staticmethod
    def define_features(genotype: Genotype = DARTS_genotype,
                        C: int = 36, layer: int = 20,
                        dropout_p: float = 0.2, **kwargs) -> FeatureExtractor:
        return FeatureExtractor(genotype, C, layer, dropout_p)

    def get_fm(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.normalize(x))[0]


class DARTS(ImageModel):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        super().add_argument(group)
        group.add_argument('--auxiliary', dest='auxiliary', action='store_true',
                           help='enable auxiliary classifier during training.')
        group.add_argument('--auxiliary_weight', dest='auxiliary_weight', type=float,
                           help='enable auxiliary classifier during training.')

    def __init__(self, name: str = 'darts', layer: int = 20,
                 auxiliary: bool = False, auxiliary_weight: float = 0.4,
                 genotype: Genotype = DARTS_genotype, model_class: type[_DARTS] = _DARTS, **kwargs):
        # TODO: ImageNet parameter settings
        super().__init__(name=name, layer=layer, genotype=genotype, model_class=model_class,
                         auxiliary=auxiliary, **kwargs)
        self._model: _DARTS
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.param_list['darts'] = ['auxiliary', 'auxiliary_weight']

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if self.auxiliary:
            isinstance(self._model.auxiliary_head, AuxiliaryHead)
            feats, feats_aux = self._model.features.forward(self._model.normalize(_input), auxiliary=True)
            logits = self._model.classifier(self._model.flatten(self._model.pool(feats)))
            logits_aux = self._model.auxiliary_head(feats_aux)
            return super().loss(_output=logits, _label=_label) \
                + self.auxiliary_weight * super().loss(_output=logits_aux, _label=_label)
        else:
            return super().loss(_input, _label, _output, *kwargs)

    def get_official_weights(self, dataset='cifar10', auxiliary: bool = False,
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        assert str(self._model.features.genotype) == str(DARTS_genotype)
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


class DARTS_Robust(DARTS):
    def __init__(self, name: str = 'darts_robust', genotype: Genotype = ROBUST_DARTS, **kwargs):
        super().__init__(name=name, genotype=genotype, **kwargs)