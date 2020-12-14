# -*- coding: utf-8 -*-


import trojanzoo.environ
import trojanzoo.dataset
import trojanzoo.model
import trojanzoo.trainer
import trojanzoo.attack

from trojanzoo.environ import env
from trojanzoo.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanzoo.environ.add_argument(parser)
    trojanzoo.dataset.add_argument(parser)
    trojanzoo.model.add_argument(parser)
    trojanzoo.trainer.add_argument(parser)
    trojanzoo.attack.add_argument(parser)
    args = parser.parse_args()

    trojanzoo.environ.create(**args.__dict__)
    dataset = trojanzoo.dataset.create(**args.__dict__)
    model = trojanzoo.model.create(dataset=dataset, **args.__dict__)
    trainer = trojanzoo.trainer.create(dataset=dataset, model=model, **args.__dict__)
    attack = trojanzoo.attack.create(dataset=dataset, model=model, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, train=trainer, attack=attack)
    attack.attack(**trainer)
