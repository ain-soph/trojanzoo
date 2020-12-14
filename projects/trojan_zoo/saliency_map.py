# -*- coding: utf-8 -*-

# CUDA_VISIBLE_DEVICES=0 python saliency_map.py --dataset sample_imagenet --width 3 --height 3 --verbose --pretrain --attack badnet

from trojanzoo.attacks import poison
from typing import List
from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack
from trojanzoo.datasets import ImageSet
from trojanzoo.models import ImageModel
from trojanzoo.attacks import BadNet

from trojanzoo.mark import Watermark
from trojanzoo.utils.tensor import save_tensor_as_img, to_numpy, save_numpy_as_img

import torch
import numpy as np
import cv2
import os
import sys

import warnings
warnings.filterwarnings("ignore")


def mix_cam(img_list: np.ndarray, mask_list: np.ndarray, alpha=0.5) -> np.ndarray:
    cam_list = []
    for img, mask in zip(img_list, mask_list):
        heatmap = cv2.applyColorMap(np.uint8(255 * to_numpy(mask)), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        heatmap = np.transpose(heatmap, (2, 0, 1))

        cam = alpha * heatmap + (1 - alpha) * to_numpy(img)
        cam_list.append(cam)
    return np.array(cam_list)


if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack())
    # sys.argv.append('--attack')
    # sys.argv.append('badnet')
    # parser.parse_args(['--dataset', 'sample_imagenet', '--width', '3', '--height', '3', 'mark_alpha', '0.0',
    #                    '--pretrain', '--attack', 'badnet'])
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']

    attack.validate_func()

    chooseset = dataset.get_dataset(mode='train')
    subset, _ = dataset.split_set(chooseset, length=1)
    clean_loader = dataset.get_dataloader(mode='train', dataset=subset)
    data = next(iter(torch.utils.data.DataLoader(subset, batch_size=len(subset), num_workers=0)))
    _input, _label = model.get_data(data)

    # _input: torch.FloatTensor = None
    # _label: torch.LongTensor = None
    # for data in dataset.loader['valid']:
    #     _input, _label = model.get_data(data)
    #     idx1 = _label != attack.target_class
    #     _input = _input[idx1]
    #     _label = _label[idx1]
    #     if len(_input) == 0:
    #         continue
    #     _class = model.get_class(_input)
    #     idx2 = _class == _label
    #     _input = _input[idx2]
    #     _label = _label[idx2]
    #     if len(_input) > len(idx1) / 2:
    #         break
    poison_input = attack.add_mark(_input)
    for i in range(len(_input)):
        save_tensor_as_img(f'./result/examples/{attack.name}/{i:0>3d}_clean.png', _input[i])
        save_tensor_as_img(f'./result/examples/{attack.name}/{i:0>3d}_poison.png', poison_input[i])

    def func(prefix: str = 'pretrain', folder_path='./result/examples/'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # _class = model.get_class(_input)
        # _conf = model.get_target_prob(_input, _class)
        # poison_class = model.get_class(poison_input)
        # poison_conf = model.get_target_prob(poison_input, poison_class)

        saliency_maps = {
            'clean': {
                'org': model.grad_cam(_input, _label),
                'target': model.grad_cam(_input, attack.target_class),
            },
            'poison': {
                'org': model.grad_cam(poison_input, _label),
                'target': model.grad_cam(poison_input, attack.target_class),
            },
        }
        norm_maps = {
            'clean': {
                'org': torch.tensor(saliency_maps['clean']['org']).flatten(1).norm(dim=1, p=1),
                'target': torch.tensor(saliency_maps['clean']['target']).flatten(1).norm(dim=1, p=1),
            },
            'poison': {
                'org': torch.tensor(saliency_maps['poison']['org']).flatten(1).norm(dim=1, p=1),
                'target': torch.tensor(saliency_maps['poison']['target']).flatten(1).norm(dim=1, p=1),
            },
        }
        mixed_maps = {
            'clean': {
                'org': mix_cam(_input, saliency_maps['clean']['org']),
                'target': mix_cam(_input, saliency_maps['clean']['target']),
            },
            'poison': {
                'org': mix_cam(_input, saliency_maps['poison']['org']),
                'target': mix_cam(_input, saliency_maps['poison']['target']),
            },
        }

        for i in range(len(_input)):
            # print("groundtruth label:                  ", _label)
            # print(f"predicted label on benign image:   {_class:3d} Confidence: {_conf:.3f}", )
            # print(f"predicted label on backdoor image: {poison_class:3d} Confidence: {poison_conf:.3f}", )
            for input_mode in ['clean', 'poison']:
                for class_mode in ['org', 'target']:
                    norm = float(norm_maps[input_mode][class_mode][i])
                    mixed_map = mixed_maps[input_mode][class_mode][i]
                    save_numpy_as_img(f'{folder_path}{i:0>3d}_{prefix}_{input_mode}_{class_mode}_{norm:.3f}.png',
                                      mixed_map)
    func(prefix='pretrain', folder_path=f'./result/examples/{attack.name}/')
    attack.load()
    func(prefix='malicious', folder_path=f'./result/examples/{attack.name}/')
