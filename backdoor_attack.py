# -*- coding: utf-8 -*-

# python backdoor_attack.py --attack badnet --verbose --pretrain --validate_interval 1 --epoch 50 --lr 1e-2 --height 3 --width 3 --mark_alpha 0.0

# python backdoor_attack.py --attack hidden_trigger --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --poison_iteration 1000 -d cifar10 -m resnet18 --lr_scheduler --step_size 20 --iteration 100 --parameters classifier
# Validate Clean:           Loss: 0.6734,          Top1 Acc: 87.630,       Top5 Acc: 99.130,       Time: 0:00:08
# Validate Trigger Tgt:     Loss: 10.5068,         Top1 Acc: 10.000,       Top5 Acc: 46.110,       Time: 0:00:01
# Validate Trigger Org:     Loss: 1.3806,          Top1 Acc: 74.590,       Top5 Acc: 97.860,       Time: 0:00:01
#
# Epoch: [  59 / 100 ]      Loss: 0.3196,          Top1 Acc: 88.317,       Top5 Acc: 100.000,      Time: 0:00:00
# Validate Clean:           Loss: 0.9571,          Top1 Acc: 80.090,       Top5 Acc: 98.410,       Time: 0:00:02
# Validate Trigger Tgt:     Loss: 1.3579,          Top1 Acc: 42.190,       Top5 Acc: 100.000,      Time: 0:00:02
# Validate Trigger Org:     Loss: 1.8767,          Top1 Acc: 60.840,       Top5 Acc: 96.680,       Time: 0:00:02

# python backdoor_attack.py --attack clean_label --mark_alpha 0.0 --height 3 --width 3 --percent 0.01 --verbose --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --save --poison_generation_method pgd
# python backdoor_attack.py --attack clean_label --mark_alpha 0.0 --height 3 --width 3 --percent 0.01 --verbose --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --save --poison_generation_method gan --train_gan
# -*- coding: utf-8 -*-

import trojanzoo.environ
import trojanzoo.dataset
import trojanzoo.model
import trojanzoo.train
import trojanzoo.mark
import trojanzoo.attack
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.train import Train
from trojanzoo.mark import Watermark
from trojanzoo.attack import BadNet

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

    args, _ = parser.parse_known_args()

    trojanzoo.environ.create(**args.__dict__)
    dataset: Dataset = trojanzoo.dataset.create(**args.__dict__)
    model: Model = trojanzoo.model.create(dataset=dataset, **args.__dict__)
    optimizer, lr_scheduler, train_args = trojanzoo.train.create(dataset=dataset, model=model, **args.__dict__)
    mark: Watermark = trojanzoo.mark.create(dataset=dataset, **args.__dict__)
    attack: BadNet = trojanzoo.attack.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, mark=mark, train=Train, attack=attack)
    attack.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
