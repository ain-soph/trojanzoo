#!/usr/bin/env python3

# python backdoor_unlearn.py --attack badnet --defense neural_cleanse --percent 0.01 --validate_interval 1 --epoch 50 --lr 1e-2 --mark_height 3 --mark_width 3 --mark_alpha 0.0

import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
import trojanvision.trainer
import trojanvision.marks
import trojanvision.attacks
import trojanvision.defenses

from trojanvision.utils import summary
import argparse
import os

from trojanvision.attacks import BadNet, Unlearn
from trojanvision.defenses import NeuralCleanse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    trojanvision.defenses.add_argument(parser)
    args, unknown = parser.parse_known_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **args.__dict__)
    defense: NeuralCleanse = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)

    simple_parser = argparse.ArgumentParser()
    simple_parser.add_argument('--mark_source', dest='mark_source', type=str, default='defense')
    args, unknown = simple_parser.parse_known_args()
    mark_source: bool = args.mark_source

    if mark_source == 'attack':
        mark_source = attack.name
    elif mark_source in ['defense', defense.name]:
        mark_source = f'{attack.name} {defense.name}'

    if mark_source == attack.name:
        attack.load()
    elif mark_source == f'{attack.name} {defense.name}':
        attack.load()
        defense.load()
    else:
        raise Exception(mark_source)

    atk_unlearn: Unlearn = trojanvision.attacks.create(mark=mark, target_class=attack.target_class, percent=attack.target_class,
                                                       mark_source=mark_source,
                                                       dataset=dataset, model=model, attack_name='unlearn')

    # ------------------------------------------------------------------------ #
    atk_unlearn.attack(**trainer)
    atk_unlearn.save()
    attack.mark.load_npz(os.path.join(attack.folder_path, attack.get_filename() + '.npz'))
    attack.validate_fn()
