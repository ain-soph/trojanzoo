#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python eval.py --verbose 1 --color --lr 0.01 --validate_interval 0 --dataset mnist --batch_size 256 --weight_decay 5e-4 --epoch 300

# https://github.com/VICO-UoE/DatasetCondensation

import trojanvision
from trojanvision.utils import summary
import argparse

import torch
from torch.utils.data import TensorDataset
from trojanzoo.utils.logger import SmoothedValue
from trojanvision.utils.model import weight_init
from utils import augment, get_daparam

from model import ConvNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    parser.add_argument('--num_eval', type=int, default=5)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    args = parser.parse_args()

    num_eval: int = args.num_eval
    epoch_eval_train: int = args.epoch_eval_train

    trojanvision.models.class_dict['convnet'] = ConvNet

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)

    eval_model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    eval_trainer = trojanvision.trainer.create(dataset=dataset, model=eval_model, **args.__dict__)
    eval_train_args = dict(**eval_trainer)
    eval_train_args['epoch'] = epoch_eval_train

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
        if dataset.name == 'mnist':
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
        _, acc = eval_model._validate(verbose=False)
        accs.update(acc)
    print(accs)
