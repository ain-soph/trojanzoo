#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=2 python eval_combine.py --verbose 1 --color --lr 0.01 --dataset cifar10 --batch_size 256 --epoch_eval_train 100 --file_path1 ./adv_train.pt --file_path2 ./normal.pt --validate_interval 0 --model convnet

# https://github.com/VICO-UoE/DatasetCondensation

import trojanvision
from trojanvision.utils import summary
import argparse

import torch
from torch.utils.data import TensorDataset
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
    parser.add_argument('--file_path1', required=True)
    parser.add_argument('--file_path2', required=True)
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
    # eval_train_args['adv_train'] = False

    if env['verbose']:
        summary(env=env, dataset=dataset, model=eval_model, trainer=eval_trainer)

    result1 = torch.load(args.file_path1)
    result2 = torch.load(args.file_path2)
    print('file_path1: ', args.file_path1)
    print('file_path2: ', args.file_path2)
    image_syn1, label_syn1 = result1['image_syn'], result1['label_syn']
    image_syn2, label_syn2 = result2['image_syn'], result2['label_syn']
    image_syn = torch.cat([image_syn1, image_syn2], dim=1)
    label_syn = torch.cat([label_syn1, label_syn2], dim=1)
    # image_syn: torch.Tensor = image_syn1
    # label_syn: torch.Tensor = label_syn1

    param_augment = get_daparam(dataset.name)

    def get_data_fn(data, **kwargs):
        _input, _label = eval_model.get_data(data, **kwargs)
        _input = augment(_input, param_augment, device=env['device'])
        return _input, _label

    for _ in range(num_eval):
        weight_init(eval_model._model)
        dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                      label_syn.detach().clone().flatten(0, 1))
        loader_train = dataset.get_dataloader(mode='train', pin_memory=False, num_workers=0,
                                              dataset=dst_syn_train)
        eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn, **eval_train_args)
        eval_model._validate()
