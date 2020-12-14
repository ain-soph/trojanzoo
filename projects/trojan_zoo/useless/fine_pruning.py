from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq
from trojanzoo.parser import Parser_Mark
from trojanzoo.parser.attack import Parser_BadNet

from trojanzoo.datasets import ImageSet
from trojanzoo.models import ImageModel
from trojanzoo.mark import Watermark
from trojanzoo.attacks.backdoor.hidden_trigger import Hidden_Trigger
from trojanzoo.defenses.backdoor import Neural_Cleanse
from trojanzoo.defenses.backdoor import Fine_Pruning
from trojanzoo.utils import normalize_mad

import argparse

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_BadNet())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: Hidden_Trigger = parser.module_list['attack']

    attack.load(epoch=train_args['epoch'])
    attack.validate_func()

    # ------------------------------------------------------------------------ #

    defense: Fine_Pruning = Fine_Pruning(dataset=dataset, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    defense.detect()


# python fine_pruning.py --attack badnet --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --epoch 1
