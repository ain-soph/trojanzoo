#!/usr/bin/env python3

from trojanvision.configs import Config, config
import trojanzoo.trainer

from typing import TYPE_CHECKING
from trojanvision.datasets import ImageSet    # TODO: python 3.10
from trojanvision.models import ImageModel
import argparse
if TYPE_CHECKING:
    pass


class Trainer(trojanzoo.trainer.Trainer):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        super().add_argument(group)
        group.add_argument('--adv_train', dest='adv_train', action='store_true',
                           help='enable adversarial training.')
        group.add_argument('--adv_train_alpha', dest='adv_train_alpha', type=float,
                           help='adversarial training PGD alpha, defaults to 2/255.')
        group.add_argument('--adv_train_epsilon', dest='adv_train_epsilon', type=float,
                           help='adversarial training PGD epsilon, defaults to 8/255.')
        group.add_argument('--adv_train_iter', dest='adv_train_iter', type=int,
                           help='adversarial training PGD iteration, defaults to 7.')


def add_argument(parser: argparse.ArgumentParser, ClassType: type[Trainer] = Trainer) -> argparse._ArgumentGroup:
    return trojanzoo.trainer.add_argument(parser=parser, ClassType=ClassType)


def create(dataset_name: str = None, dataset: ImageSet = None, model: ImageModel = None,
           config: Config = config, ClassType=Trainer, **kwargs):
    return trojanzoo.trainer.create(dataset_name=dataset_name, dataset=dataset,
                                    ClassType=ClassType,
                                    model=model, config=config, **kwargs)
