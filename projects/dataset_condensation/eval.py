#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python eval.py --verbose 1 --color --lr 0.01 --validate_interval 0 --dataset mnist --batch_size 256 --weight_decay 5e-4 --lr_scheduler --epoch 300 --dataset_normalize

# https://github.com/VICO-UoE/DatasetCondensation

from typing import OrderedDict
import trojanvision
from trojanvision.utils import summary
import argparse

import torch
from torch.utils.data import TensorDataset
from trojanzoo.utils.logger import SmoothedValue
from trojanvision.utils.model import weight_init
from utils import match_loss, augment, get_daparam

from model import ConvNet

num_eval = 5
epoch_eval_train = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    trojanvision.models.class_dict['convnet'] = ConvNet

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)

    eval_model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    eval_trainer = trojanvision.trainer.create(dataset=dataset, model=eval_model, **args.__dict__)
    eval_train_args = dict(**eval_trainer)
    eval_train_args['epoch'] = epoch_eval_train // 2

    if env['verbose']:
        summary(env=env, dataset=dataset, model=eval_model, trainer=eval_trainer)

    # a: OrderedDict = torch.load('./result.pth')
    # b = OrderedDict()
    # for key, value in a.items():
    #     key: str
    #     b[key.replace('classifier', 'classifier.fc')] = value

    # eval_model._model.load_state_dict(b)
    # eval_model._validate()

    result = torch.load('./result.pt')
    image_syn, label_syn = result['data'][0]

    param_augment = get_daparam(dataset.name)

    def get_data_fn(data, **kwargs):
        _input, _label = dataset.get_data(data, **kwargs)
        _input = augment(_input, param_augment, device=env['device'])
        return _input, _label
    accs = SmoothedValue(fmt='{global_avg:.3f} ({min:.3f}  {max:.3f})')
    for _ in range(num_eval):
        weight_init(eval_model._model)
        dst_syn_train = TensorDataset(image_syn.detach().clone(),
                                      label_syn.detach().clone())
        loader_train = dataset.get_dataloader(mode='train', pin_memory=False, num_workers=0,
                                              dataset=dst_syn_train)
        eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn, **eval_train_args)
        eval_train_args['optimizer'].param_groups[0]['lr'] *= 0.1
        eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn, **eval_train_args)
        _, acc = eval_model._validate(verbose=False)
        accs.update(acc)
    print(accs)
