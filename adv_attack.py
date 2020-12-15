# -*- coding: utf-8 -*-


import trojanzoo.environ
import trojanzoo.datasets
import trojanzoo.models
import trojanzoo.trainer
import trojanzoo.attacks

from trojanzoo.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanzoo.environ.add_argument(parser)
    trojanzoo.datasets.add_argument(parser)
    trojanzoo.models.add_argument(parser)
    trojanzoo.trainer.add_argument(parser)
    trojanzoo.attacks.add_argument(parser)
    args = parser.parse_args()

    env = trojanzoo.environ.create(**args.__dict__)
    dataset = trojanzoo.datasets.create(**args.__dict__)
    model = trojanzoo.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanzoo.trainer.create(dataset=dataset, model=model, **args.__dict__)
    attack = trojanzoo.attacks.create(dataset=dataset, model=model, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, train=trainer, attack=attack)
    attack.attack(**trainer)
