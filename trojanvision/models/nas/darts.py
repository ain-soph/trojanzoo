#!/usr/bin/env python3
from trojanvision.datasets import ImageSet
from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs.darts import FeatureExtractor, AuxiliaryHead, Genotype
from trojanvision.utils.model_archs.darts import genotypes

import torch
import torch.nn as nn
from torchvision.datasets.utils import download_file_from_google_drive
import os
from collections import OrderedDict

from typing import TYPE_CHECKING
import argparse  # TODO: python 3.10
if TYPE_CHECKING:
    pass

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
        self.auxiliary_head: nn.Sequential = None
        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C=self.features.aux_dim, num_classes=self.num_classes)

    @staticmethod
    def define_features(genotype: Genotype = genotypes.darts,
                        C: int = 36, layers: int = 20,
                        dropout_p: float = 0.2, **kwargs) -> FeatureExtractor:
        return FeatureExtractor(genotype, C, layers, dropout_p, **kwargs)


class DARTS(ImageModel):
    available_models = ['darts']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--model_arch', help='Model Architecture (genotype name), defaults to be "darts"')
        group.add_argument('--auxiliary', action='store_true', help='enable auxiliary classifier during training.')
        group.add_argument('--auxiliary_weight', type=float, help='weight for auxiliary loss, defaults to be 0.4')
        return group

    def __init__(self, name: str = 'darts', model_arch: str = 'darts',
                 layers: int = 20, C: int = 36, dropout_p: float = 0.2,
                 auxiliary: bool = False, auxiliary_weight: float = 0.4,
                 genotype: Genotype = None, model: type[_DARTS] = _DARTS, **kwargs):
        # TODO: ImageNet parameter settings
        if genotype is None:
            model_arch = model_arch.lower()
            name = model_arch
            try:
                genotype = getattr(genotypes, model_arch)
            except AttributeError as e:
                print('Available Model Architectures: ')
                model_arch_list = [element for element in dir(genotypes) if '__' not in element and
                                   element not in ['Genotype', 'PRIMITIVES', 'namedtuple']]
                print(model_arch_list)
                raise e
        self.layers = layers
        self.C = C
        self.dropout_p = dropout_p
        self.genotype = genotype
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        super().__init__(name=name, layers=layers, C=C, dropout_p=dropout_p,
                         genotype=genotype, model=model,
                         auxiliary=auxiliary, **kwargs)
        self._model: _DARTS
        self.param_list['darts'] = ['layers', 'C', 'dropout_p', 'genotype']
        if auxiliary:
            self.param_list['darts'].insert(0, 'auxiliary_weight')

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, amp: bool = False, **kwargs) -> torch.Tensor:
        if self.auxiliary:
            assert isinstance(self._model.auxiliary_head, nn.Sequential)
            if amp:
                with torch.cuda.amp.autocast():
                    return self.loss_with_aux(_input, _label)
            return self.loss_with_aux(_input, _label)
        else:
            return super().loss(_input, _label, _output, **kwargs)

    def loss_with_aux(self, _input: torch.Tensor = None, _label: torch.Tensor = None) -> torch.Tensor:
        feats, feats_aux = self._model.features.forward_with_aux(self._model.normalize(_input))
        logits: torch.Tensor = self._model.classifier(self._model.flatten(self._model.pool(feats)))
        logits_aux: torch.Tensor = self._model.auxiliary_head(feats_aux)
        return super().loss(_output=logits, _label=_label) \
            + self.auxiliary_weight * super().loss(_output=logits_aux, _label=_label)

    def load(self, *args, strict: bool = False, **kwargs):
        return super().load(*args, strict=strict, **kwargs)

    def get_official_weights(self, dataset: str = None, **kwargs) -> OrderedDict[str, torch.Tensor]:
        assert str(self.genotype) == str(genotypes.darts)
        if dataset is None and isinstance(self.dataset, ImageSet):
            dataset = self.dataset.name
        file_name = f'darts_{dataset}.pt'
        folder_path = os.path.join(torch.hub.get_dir(), 'darts')
        download_file_from_google_drive(file_id=url[dataset], root=folder_path, filename=file_name)
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
            if 'auxiliary_head' not in new_keys[i] and 'auxiliary_head' in old_keys[j]:
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
