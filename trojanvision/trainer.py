#!/usr/bin/env python3

from trojanvision.configs import Config, config
from trojanzoo.trainer import Trainer
import trojanzoo.trainer

from typing import TYPE_CHECKING
from trojanvision.datasets import ImageSet    # TODO: python 3.10
from trojanvision.models import ImageModel
import argparse
if TYPE_CHECKING:
    pass


def add_argument(parser: argparse.ArgumentParser, ClassType: type[Trainer] = Trainer) -> argparse._ArgumentGroup:
    return trojanzoo.trainer.add_argument(parser=parser, ClassType=ClassType)


def create(dataset_name: str = None, dataset: ImageSet = None, model: ImageModel = None,
           config: Config = config, ClassType=Trainer, **kwargs):
    return trojanzoo.trainer.create(dataset_name=dataset_name, dataset=dataset,
                                    ClassType=ClassType,
                                    model=model, config=config, **kwargs)
