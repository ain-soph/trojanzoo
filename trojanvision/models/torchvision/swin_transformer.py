#!/usr/bin/env python3

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.swin_transformer import (Swin_T_Weights, Swin_S_Weights, Swin_B_Weights,
                                                 Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights)
from collections import OrderedDict

from torchvision.models.swin_transformer import _swin_transformer, SwinTransformerBlockV2, PatchMergingV2
from typing import Optional, Any


class _SwinTransformer(_ImageModel):

    def __init__(self, name: str = 'swin_v2_t', **kwargs):
        super().__init__(**kwargs)
        if 'comp' in name:
            ModelClass = eval(name)
        else:
            ModelClass = getattr(torchvision.models, name)
        _model: torchvision.models.SwinTransformer = ModelClass(num_classes=self.num_classes)

        self.features = _model.features
        self.features.add_module('norm', _model.norm)
        self.features.add_module('permute', _model.permute)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.head),
        ]))


class SwinTransformer(ImageModel):
    available_models = {'swin_t', 'swin_s', 'swin_b',
                        'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
                        'swin_t_comp', 'swin_v2_t_comp',
                        }
    weights = {
        'swin_t': Swin_T_Weights,
        'swin_s': Swin_S_Weights,
        'swin_b': Swin_B_Weights,
        'swin_v2_t': Swin_V2_T_Weights,
        'swin_v2_s': Swin_V2_S_Weights,
        'swin_v2_b': Swin_V2_B_Weights,
    }

    def __init__(self, name: str = 'swin_v2_t',
                 model: type[_SwinTransformer] = _SwinTransformer, **kwargs):
        super().__init__(name=name, model=model, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        _dict['features.norm.weight'] = _dict['norm.weight']
        _dict['features.norm.bias'] = _dict['norm.bias']
        del _dict['norm.weight']
        del _dict['norm.bias']
        _dict['classifier.fc.weight'] = _dict['head.weight']
        _dict['classifier.fc.bias'] = _dict['head.bias']
        del _dict['head.weight']
        del _dict['head.bias']
        return _dict


def swin_t_comp(*, weights: Optional[Swin_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    weights = Swin_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[2, 2],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 4],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def swin_v2_t_comp(*, weights: Optional[Swin_V2_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    weights = Swin_V2_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[2, 2],
        embed_dim=96,
        depths=[2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 4],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
