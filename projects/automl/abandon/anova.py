#!/usr/bin/env python3

import torch
import numpy as np
from trojanzoo.utils.data import dataset_to_list

import trojanvision.environ
import trojanvision.datasets
import trojanvision.models

from trojanvision.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")


def get_feats(_input: torch.Tensor, model: trojanvision.models.ImageModel):
    input_list: list[torch.Tensor] = torch.tensor_split(_input, _input.size(0) // dataset.batch_size)
    feats = torch.cat([model.get_final_fm(sub_input) for sub_input in input_list])
    return feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)
    model._validate()

    bmss_list: list[float] = []
    wmss_list: list[float] = []
    f_list: list[float] = []
    print()
    print()
    for c in range(dataset.num_classes):
        print('class: ', c)
        train_data, _ = dataset_to_list(dataset.get_dataset(mode='train', class_list=[c]))
        valid_data, _ = dataset_to_list(dataset.get_dataset(mode='valid', class_list=[c]))
        train_data, valid_data = torch.stack(train_data), torch.stack(valid_data)
        train_data, valid_data = train_data.to(env['device']), valid_data.to(env['device'])
        train_feats, valid_feats = get_feats(train_data, model), get_feats(valid_data, model)
        Y_total = torch.cat([train_feats, valid_feats]).mean(dim=0)
        bmss = train_feats.size(0) * (train_feats.mean(dim=0) - Y_total).pow(2).sum(dim=-1) \
            + valid_feats.size(0) * (valid_feats.mean(dim=0) - Y_total).pow(2).sum(dim=-1)
        wmss = ((train_feats - train_feats.mean(dim=0)).pow(2).sum(dim=-1).sum()
                + (valid_feats - valid_feats.mean(dim=0)).pow(2).sum(dim=-1).sum()) \
            / (train_feats.size(0) + valid_feats.size(0) - 2)
        f_score = bmss / wmss
        bmss_list.append(float(bmss))
        wmss_list.append(float(wmss))
        f_list.append(float(f_score))
        print('    BMSS: ', float(bmss))
        print('    WMSS: ', float(wmss))
        print('    F:    ', float(f_score))
        print('-' * 20)
        print()
    print(f'BMSS: {np.mean(bmss_list):7.3f} ({np.min(bmss_list):7.3f}, {np.max(bmss_list):7.3f})')
    print(f'WMSS: {np.mean(wmss_list):7.3f} ({np.min(wmss_list):7.3f}, {np.max(wmss_list):7.3f})')
    print(f'F:    {np.mean(f_list):7.3f} ({np.min(f_list):7.3f}, {np.max(f_list):7.3f})')
