# -*- coding: utf-8 -*-

import trojanzoo.attack
import trojanzoo.mark
import trojanzoo.model
import trojanzoo.dataset
import trojanzoo.environ
from trojanzoo.environ import env
from trojanzoo.utils import summary
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.mark import Watermark
from trojanzoo.attack import BadNet

import argparse

from trojanzoo.defense import Defense_Backdoor

from trojanzoo.utils.model import AverageMeter

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']


# -*- coding: utf-8 -*-


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanzoo.utils.environ.add_argument(parser)
    trojanzoo.dataset.add_argument(parser)
    trojanzoo.model.add_argument(parser)
    trojanzoo.utils.mark.Watermark.add_argument(parser)
    trojanzoo.add_argument(parser)

    args = parser.parse_args()

    trojanzoo.utils.environ.create(**args.__dict__)
    dataset: Dataset = trojanzoo.dataset.create(**args.__dict__)
    model: Model = trojanzoo.model.create(dataset=dataset, **args.__dict__)
    attack: BadNet = trojanzoo.attack.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, mark=mark, attack=attack)

    attack.load()
    attack.validate_func()
