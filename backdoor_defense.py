# -*- coding: utf-8 -*-

# python backdoor_defense.py --attack badnet --defense neural_cleanse --verbose --pretrain --validate_interval 1 --epoch 50 --lr 1e-2 --mark_height 3 --mark_width 3 --mark_alpha 0.0

import trojanzoo.environ
import trojanzoo.dataset
import trojanzoo.model
import trojanzoo.train
import trojanzoo.mark
import trojanzoo.attack
import trojanzoo.defense
from trojanzoo.train import Train
from trojanzoo.attack import BadNet
from trojanzoo.defense import Defense_Backdoor

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
    trojanzoo.train.add_argument(parser)
    trojanzoo.mark.add_argument(parser)
    trojanzoo.attack.add_argument(parser)
    trojanzoo.defense.add_argument(parser)

    args, _ = parser.parse_known_args()

    trojanzoo.environ.create(**args.__dict__)
    dataset = trojanzoo.dataset.create(**args.__dict__)
    model = trojanzoo.model.create(dataset=dataset, **args.__dict__)
    optimizer, lr_scheduler, train_args = trojanzoo.train.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanzoo.mark.create(dataset=dataset, **args.__dict__)
    attack: BadNet = trojanzoo.attack.create(dataset=dataset, model=model, mark=mark, **args.__dict__)
    defense: Defense_Backdoor = trojanzoo.defense.create(dataset=dataset, model=model, attack=attack, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, mark=mark, train=Train, attack=attack, defense=defense)
    defense.detect(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
