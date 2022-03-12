#!/usr/bin/env python3

r"""It is equivalent to :ref:`trojanzoo.trainer <trojanzoo.trainer>`.

Note:
    The only difference is that it uses ``trojanvision.configs.config``
    as the default parameter passed to :func:`trojanzoo.trainer.create()`.
"""

from trojanvision.configs import config
from trojanzoo.trainer import Trainer
import trojanzoo.trainer

from typing import TYPE_CHECKING
from trojanvision.datasets import ImageSet    # TODO: python 3.10
from trojanvision.models import ImageModel
from trojanzoo.configs import Config
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
