# -*- coding: utf-8 -*-

# python backdoor_attack.py --attack badnet --verbose --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --height 1 --width 1
# python backdoor_attack.py --attack trojannn --verbose --pretrain --validate_interval 1 --mark_ratio 0.3
# python backdoor_attack.py --attack latent_backdoor --verbose --pretrain --validate_interval 1 --mark_ratio 0.3
# python backdoor_attack.py --attack clean_label --verbose --pretrain --validate_interval 1 --mark_ratio 0.3

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

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.attack import BadNet
from trojanzoo.utils.mark import Watermark

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

    # ------------------------------------------------------------------------ #
    attack.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
