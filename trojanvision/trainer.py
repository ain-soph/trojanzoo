#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanvision.configs import Config, config
from trojanvision.datasets import ImageSet
from trojanvision.models import ImageModel
import trojanzoo.trainer
from trojanzoo.trainer import Trainer, add_argument


def create(dataset_name: str = None, dataset: ImageSet = None, model: ImageModel = None,
           config: Config = config, **kwargs):
    return trojanzoo.trainer.create(dataset_name=dataset_name, dataset=dataset,
                                    model=model, config=config, **kwargs)
