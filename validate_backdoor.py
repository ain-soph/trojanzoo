# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
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

    attack.load()
    # ------------------------------------------------------------------------ #
    confidence = AverageMeter('Confidence', ':.4e')
    for data in dataset.loader['valid']:
        _input, _label = model.get_data(data)
        idx1 = _label != attack.target_class
        _input = _input[idx1]
        _label = _label[idx1]
        poison_input = attack.add_mark(_input)
        poison_label = model.get_class(poison_input)
        idx2 = poison_label == attack.target_class
        poison_input = poison_input[idx2]
        batch_conf = model.get_prob(poison_input)[:, attack.target_class].mean()
        confidence.update(batch_conf, len(poison_input))
    print(f'Confidence: {confidence.avg}')
