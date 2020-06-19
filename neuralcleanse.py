# -*- coding: utf-8 -*-

# python neuralcleanse.py --verbose --pretrain --validate_interval 1 --mark_ratio 0.3 --epoch 1

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq
from trojanzoo.parser import Parser_Mark
from trojanzoo.parser.attack import Parser_BadNet

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor.hiddentrigger import HiddenTrigger
from trojanzoo.defense.neural_cleanse import Neural_Cleanse

from trojanzoo.utils import normalize_mad

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
    attack: HiddenTrigger = parser.module_list['attack']

    attack.load(epoch=train_args['epoch'])

    # ------------------------------------------------------------------------ #

    data_shape = [dataset.n_channel]
    data_shape.extend(dataset.n_dim)
    defense: Neural_Cleanse = Neural_Cleanse(dataset=dataset, model=model, data_shape=data_shape)

    mark_list, mask_list, loss_ce_list = defense.get_potential_triggers()
    mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1)

    print('mask_norms: ', normalize_mad(mask_norms))
    print('loss: ', normalize_mad(loss_ce_list))
