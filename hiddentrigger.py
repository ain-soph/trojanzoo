# -*- coding: utf-8 -*-

# python hiddentrigger.py --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --poison_iteration 1000 --poison_num 200 -d cifar10 -m resnet18 --lr_scheduler --step_size 20 --iteration 100 --parameters classifier
# Validate Clean:           Loss: 0.6734,          Top1 Acc: 87.630,       Top5 Acc: 99.130,       Time: 0:00:08
# Validate Trigger Tgt:     Loss: 10.5068,         Top1 Acc: 10.000,       Top5 Acc: 46.110,       Time: 0:00:01
# Validate Trigger Org:     Loss: 1.3806,          Top1 Acc: 74.590,       Top5 Acc: 97.860,       Time: 0:00:01
# 
# Epoch: [  59 / 100 ]      Loss: 0.3196,          Top1 Acc: 88.317,       Top5 Acc: 100.000,      Time: 0:00:00
# Validate Clean:           Loss: 0.9571,          Top1 Acc: 80.090,       Top5 Acc: 98.410,       Time: 0:00:02
# Validate Trigger Tgt:     Loss: 1.3579,          Top1 Acc: 42.190,       Top5 Acc: 100.000,      Time: 0:00:02
# Validate Trigger Org:     Loss: 1.8767,          Top1 Acc: 60.840,       Top5 Acc: 96.680,       Time: 0:00:02


from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq
from trojanzoo.parser import Parser_Mark
from trojanzoo.parser.attack import Parser_BadNet
from trojanzoo.parser.attack import Parser_HiddenTrigger

from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor.hiddentrigger import HiddenTrigger

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_HiddenTrigger())
    parser.parse_args()
    parser.get_module()

    dataset: Dataset = parser.module_list['dataset']
    model: Model = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: HiddenTrigger = parser.module_list['attack']

    del train_args['epoch']

    # ------------------------------------------------------------------------ #
    attack.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
