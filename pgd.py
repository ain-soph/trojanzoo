# -*- coding: utf-8 -*-

# todo: Output format need modifying

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Seq
from trojanzoo.parser.attack import Parser_PGD

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.attack import PGD

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_PGD())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    attack: PGD = parser.module_list['attack']

    # ------------------------------------------------------------------------ #

    testloader = dataset.loader['test']

    # model._validate()
    correct = 0
    total = 0
    total_iter = 0
    for i, data in enumerate(testloader):
        if total >= 100:
            break
        _input, _label = model.remove_misclassify(data)
        if len(_label) == 0:
            continue
        adv_input, _iter = attack.attack(_input)

        total += 1
        if _iter:
            correct += 1
            total_iter += _iter
        print('{} / {}'.format(correct, total))
        print('current iter: ', _iter)
        print('succ rate: ', float(correct) / total)
        if correct > 0:
            print('avg  iter: ', float(total_iter) / correct)
        print()
