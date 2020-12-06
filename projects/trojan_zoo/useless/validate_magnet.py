# -*- coding: utf-8 -*-

from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Seq
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.model.image.magnet import MagNet

import torch
from typing import Tuple

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model())
    parser.parse_args()
    parser.get_module()

    dataset: Dataset = parser.module_list.dataset
    model: Model = parser.module_list.model

    loss, acc1, acc5 = model._validate(full=True)

    magnet: MagNet = MagNet(dataset=dataset, pretrain=True)

    def get_data(data: Tuple[torch.Tensor, torch.LongTensor], **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
        _input, _label = model.get_data(data)
        _input = magnet(_input)
        return _input, _label

    loss, acc1, acc5 = model._validate(full=True, get_data=get_data)
