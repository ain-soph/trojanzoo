# -*- coding: utf-8 -*-

# python strip.py --attack badnet --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --epoch 1

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.defense.backdoor import STRIP

from trojanzoo.utils import normalize_mad
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

    attack.load(epoch=train_args['epoch'])
    attack.validate_func()

    # ------------------------------------------------------------------------ #

    data_shape = [dataset.n_channel]
    data_shape.extend(dataset.n_dim)
    defense: STRIP = STRIP(dataset=dataset, model=model)

    entropy = AverageMeter('entropy', fmt='.4e')
    for i, data in enumerate(dataset.loader['test']):
        _input, _label = model.get_data(data)
        entropy.update(defense.detect(_input), n=_label.size(0))
        print('{:<10d}{:<20.4f}'.format(i, entropy.avg))
