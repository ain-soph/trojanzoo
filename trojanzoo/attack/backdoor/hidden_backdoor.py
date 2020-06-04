# -*- coding: utf-8 -*-
# import sys
# sys.path.append(r'/home/nesa320/xs/Trojan-Zoo')
# print(sys.path)

from trojanzoo.attack.attack import Attack
from trojanzoo.attack.backdoor_attack import Backdoor_Attack
from trojanzoo.imports import *
from trojanzoo.utils import *
import random
import os
from PIL.Image import Image
import random
from typing import Union, List
from copy import deepcopy
from trojanzoo.utils import to_tensor, read_img_as_tensor, byte2float, repeat_to_batch, prints
from trojanzoo.utils.attack import add_mark
from trojanzoo import __file__ as root_file
root_dir = os.path.dirname(os.path.abspath(__file__))


class HiddenBackdoor(Backdoor_Attack):

    name = 'hiddenbackdoor'

    def __init__(self,
                 poisoned_image_num: int = 100,
                 poison_generation_iteration: int = 5000,
                 poison_lr: float = 0.01,
                 preprocess_layer: str = 'feature',
                 epsilon: int = 16,
                 decay: bool = False,
                 decay_iteration: int = 2000,
                 decay_ratio: float = 0.95,
                 **kwargs):
        """
        HiddenBackdoor attack is different with trojan nn(References: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech),the mark and mask is designated and stable, we continue these used in paper(References: https://arxiv.org/abs/1910.00033).

        :param poisoned_image_num: the number of poisoned images, defaults to 100
        :type poisoned_image_num: int, optional
        :param poison_generation_iteration: the iteration times used to generate one poison image, defaults to 5000
        :type poison_generation_iteration: int, optional
        :param poison_lr: the learning rate used for generating poisoned images, defaults to 0.01
        :type poison_lr: float, optional
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images, defaults to 'feature'
        :type preprocess_layer: str, optional
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image, defaults to 16
        :type epsilon: int, optional
        :param decay: specify whether the learning rate decays with iteraion times, defaults to False
        :type decay: bool, optional
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once, defaults to 2000
        :type decay_iteration: int, optional
        :param decay_ratio: specify the learning rate decay proportion, defaults to 0.95
        :type decay_ratio: float, optional
        """

        super().__init__(self, **kwargs)

        self.poisoned_image_num = poisoned_image_num
        self.poison_generation_iteration = poison_generation_iteration
        self.poison_lr = poison_lr
        self.preprocess_layer = preprocess_layer
        self.epsilon = epsilon
        self.decay = decay
        self.decay_iteration = decay_iteration
        self.decay_ratio = decay_ratio

        self.percent = float(
            self.poisoned_image_num /
            self.dataset.get_full_dataset('train').__len__().item()
        )  # update self.percent according to self.poisoned_image_num
        prints("The percent of poisoned image:{}".format(self.percent),
               indent=self.indent)

    def attack(self,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
               iteration: int = None,
               **kwargs):
        """
        Retrain the model with normal images and poisoned images whose label haven't be modified, finetune self.model in this function. self.attack function differes from Backdoor_Attack.attack funtions at get_data=self.get_data(keep_org = False)

        :param optimizer: specify the optimizer
        :type optimizer: torch.optim.Optimizer
        :param lr_scheduler: specify the lr_scheduler, dynamic learning rate
        :type lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        :param iteration: how many epoches that the model needs to retrained, defaults to None
        :type iteration: int, optional
        """
        if iteration is None:
            iteration = self.iteration
        self.model._train(epoch=iteration,
                          optimizer=optimizer,
                          lr_scheduler=lr_scheduler,
                          get_data=self.get_data(keep_org=False),
                          validate_func=self.validate_func,
                          **kwargs)

    def get_data(self,
                 data: (torch.Tensor, torch.LongTensor),
                 keep_org: bool = True) -> (torch.Tensor, torch.LongTensor):
        """
        When keep_org= True, get the normal inputs and labels.
        When keep_org= False, get the normal inputs and labels and poisoned inputs and their labels.
        :param data: the original input and label
        :type data: torch.Tensor, torch.LongTensor
        :param keep_orig: specify whether to insert poisoned inputs and labels, defaults to True
        :type keep_orig: bool, optional
        :return: _input, _label
        :rtype: torch.Tensor, torch.LongTensor
        """
        _input, _label = self.model.get_data(data)
        if not keep_org:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input)
            # source_image =  # defaults to callable
            # target_image =  # defaults to callable
            _input = self.generate_poisoned_image(source_image, target_image)
            _label = self.target_class * torch.ones_like(org_label)

            _input = torch.cat((_input, org_input))
            _label = torch.cat((_label, org_label))
        return _input, _label

    def adjust_lr(self,
                  iteration,
                  decay: bool = False,
                  decay_ratio: float = None,
                  decay_iteration: int = None) -> (float):
        """
        In the process of generating poisoned inputs, the learning rate will change with the iteration times.
        :param iteration: the number of iteration in the process of generating poisoned image
        :type iteration: int, optional
        :param decay: specify whether the learning rate decays with iteraion times, defaults to False
        :type decay: bool, optional
        :param decay_ratio: specify the learning rate decay proportion, defaults to 0.95
        :type decay_ratio: float, optional
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once, defaults to 2000
        :type decay_iteration: int, optional
        :return: lr or self.poison_lr: the computed learning rate
        :rtype: float
        """
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration

        if decay:
            lr = self.poison_lr
            lr = lr * (decay_ratio**(iteration // decay_iteration))
            return lr
        else:
            return self.poison_lr

    def generate_poisoned_image(self,
                                source_image: (torch.Tensor, torch.LongTensor),
                                target_image: (torch.Tensor, torch.LongTensor),
                                preprocess_layer: str = None,
                                poison_generation_iteration: int = None,
                                epsilon: int = None,
                                decay: bool = None,
                                decay_ratio: float = None,
                                decay_iteration: int = None,
                                **kwargs) -> (torch.Tensor):
        """
        According to the sampled target images and the sampled source images patched by the trigger ,modify the target inputs to generate poison inputs ,that is close to inputs of target category in pixel space and also close to source inputs patched by the trigger in feature space.
        :param source_image: self.poisoned_image_num source images, other than target category, sampled from train dataset
        :type source_image: torch.Tensor, torch.LongTensor, optional
        :param target_image: self.poisoned_image_num target images sampled from the images of target category in train dataset
        :type target_image: torch.Tensor, torch.LongTensor, optional
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images
        :type preprocess_layer: str, optional
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image
        :type epsilon: int, optional
        :param decay: specify whether the learning rate decays with iteraion times
        :type decay: bool, optional
        :param decay_ratio: specify the learning rate decay proportion
        :type decay_ratio: float, optional
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once
        :type decay_iteration: int, optional
        :return: generated_poisoned_input: the generated poisoned inputs
        :rtype: torch.Tensor
        """

        if preprocess_layer is None:
            preprocess_layer = self.preprocess_layer
        if poison_generation_iteration is None:
            poison_generation_iteration = self.poison_generation_iteration
        if epsilon is None:
            epsilon = self.epsilon
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration

        source_input, source_label = self.model.get_data(source_image)
        target_input, target_label = self.model.get_data(target_image)
        generated_poisoned_input = torch.zeros_like(source_input).to(
            source_image.device)

        pert = nn.Parameter(
            torch.zeros_like(target_input,
                             requires_grad=True).to(source_image.device))
        source_input = self.add_mark(source_input)

        output1 = self.model(source_input)
        feat1 = to_tensor(
            self.model.get_layer(
                source_input, layer_output=preprocess_layer)).detach().clone()
        for j in range(poison_generation_iteration):
            output2 = model(target_input + pert)
            feat2 = to_tensor(
                self.model.get_layer(
                    target_input,
                    layer_output=preprocess_layer)).detach().clone()
            feat11 = feat1.clone()
            dist = torch.cdist(feat1, feat2)
            for _ in range(feat2.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
                feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                dist[dist_min_index[0], dist_min_index[1]] = 1e5

            loss1 = ((feat1 - feat2)**2).sum(
                dim=1
            )  #  Decrease the distance between sourced images patched by trigger and target images
            loss = loss1.sum()
            losses.update(loss.item(), source_input.size(0))
            loss.backward()
            lr = self.adjust_lr(iteration=j,
                                decay=decay,
                                decay_ratio=decay_ratio,
                                decay_iteration=decay_iteration)
            pert = pert - lr * pert.grad
            pert = torch.clamp(pert, -(epsilon / 255.0),
                               epsilon / 255.0).detach_()
            pert = pert + target_input
            pert = pert.clamp(0, 1)  # restrict the pixel value range resonable
            if j % 100 == 0:
                print(
                    "Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.4f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}"
                    .format(epoch, i, j, lr, losses.val, losses.avg))
            if loss1.max().item() < 10 or j == (poison_generation_iteration -
                                                1):
                for k in range(target_input.size(0)):
                    input2_pert = (pert[k].clone())
                    generated_poisoned_input[k] = input2_pert
                break
            pert = pert - target_image
            pert.requires_grad = True
        return generated_poisoned_input


# from trojanzoo.parser.parser import Parser
# from trojanzoo.utils.loader import get_dataset
# from trojanzoo.dataset import Dataset
# from trojanzoo.model import Model
# from trojanzoo.config import Config
# from trojanzoo.utils.param import Module, Param
# from trojanzoo.imports import *
# from trojanzoo.utils import *
# from trojanzoo.attack.attack import Attack
# from trojanzoo.attack.backdoor_attack import Backdoor_Attack, watermark
# from trojanzoo.parser.attack import Parser_Perturb
# from trojanzoo.config import Config
# from trojanzoo.attack.hidden import HiddenBackdoor
# from trojanzoo.dataset import Dataset, ImageSet
# from trojanzoo.model import Model, ImageModel
# env = Config.env
# config = Config.config

# param = Param(default={'poisoned_image_num': 100, 'poison_generation_iteration': 5000, 'poison_lr': 0.01, 'preprocess_layer': 'feature', 'epsilon': 16, 'decay': False, 'decay_iteration': 2000, 'decay_ratio': 0.95})
# datashape = Param({'default':{'data_shape':[3,32,32]}, 'gtsrb':{'data_shape':[3,32,32]},'imagenet':{'data_shape':[3,224,224]}, 'default':{'data_shape':[3,224,224]}})

# # output  attention
# class Parser_Hidden_Backdoor(Parser_Perturb):
#     def __init__(self, *args, param=param, datashape = datashape, **kwargs):
#         super().__init__(*args, param=param, datashape = datashape, **kwargs)

#     @classmethod
#     def add_argument(cls, parser):
#         super().add_argument(parser)
#         parser.set_defaults(module_name='hidden_backdoor')

#         parser.add_argument('-t', '--target_class', dest='target_class',default=0, type=int)
#         parser.add_argument('--alpha', dest='alpha',default=1.0, type=float)
#         parser.add_argument('--percent', dest='percent',default=0.1, type=float)

#         parser.add_argument('--folder_path', dest='folder_path',default=env['result_dir'], type=str)
#         parser.add_argument('--iteration', dest='iteration',default=50, type=int)
#         parser.add_argument('--early_stop', dest='early_stop',default=False, action='store_true')
#         parser.add_argument('--stop_confidence', dest='stop_confidence',default=0.75)
#         parser.add_argument()
#         parser.add_argument('--indent', dest='indent',default=0)
#         parser.add_argument('--optimizer', dest='optimizer',default='SGD')
#         parser.add_argument('--lr_scheduler', dest='lr_scheduler',default=False, action='store_true')

#         parser.add_argument('--edge_color', dest='edge_color',default='black')
#         parser.add_argument('--mark_path', dest='path',default=env['data_dir']+'mark/square_white.png')
#         parser.add_argument('--mark_height', dest='height',default=0, type=int)
#         parser.add_argument('--mark_width', dest='width',default=0, type=int)
#         parser.add_argument('--mark_height_ratio', dest='height_ratio',default=1.0, type=float)
#         parser.add_argument('--mark_width_ratio', dest='width_ratio',default=1.0, type=float)
#         parser.add_argument('--mark_height_offset', dest='height_offset',default=None, type=int)
#         parser.add_argument('--mark_width_offset', dest='width_offset',default=None, type=int)

#         parser.add_argument('--poisoned_image_num', dest='poisoned_image_num',default=100, type=int)
#         parser.add_argument('--poison_generation_iteration', dest='poison_generation_iteration',default=5000, type=int)
#         parser.add_argument('--poison_lr', dest='poison_lr',default=0.01, type=float)
#         parser.add_argument('--preprocess_layer', dest='preprocess_layer',default='feature', type=str)
#         parser.add_argument('--epsilon', dest='epsilon',default=16, type=int)
#         parser.add_argument('--decay', dest='decay',default=False, type=bool)
#         parser.add_argument('--decay_iteration', dest='decay_iteration',default=200, type=int)
#         parser.add_argument('--decay_ratio', dest='decay_ratio',default=0.95, type=float)
#     def get_module(self,  **kwargs):
#

# from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Watermark, Parser_Hidden_Backdoor, Parser_Seq
# from trojanzoo.attack.attack import Attack
# from trojanzoo.attack.backdoor_attack import Backdoor_Attack, Watermark
# from trojanzoo.parser.attack import Parser_Perturb
# from trojanzoo.attack.hidden import HiddenBackdoor
# from trojanzoo.dataset import Dataset, ImageSet
# from trojanzoo.model import Model, ImageModel
# from trojanzoo.utils import prints
# import warnings
# warnings.filterwarnings("ignore")

# if __name__ == '__main__':
#     parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),Parser_Watermark(), Parser_Hidden_Backdoor())
#     parser.parse_args()
#     parser.get_module()

#     dataset: ImageSet = parser.module_list['dataset']
#     model: ImageModel = parser.module_list['model']
#     optimizer, lr_scheduler, train_args = parser.module_list['train']
#     watermark: Watermark = parser.module_list[]
#     hiddenbackdoor: HiddenBackdoor = parser.module_list[]

#     # ------------------------------------------------------------------------ #
#     hiddenbackdoor.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, dataset = dataset, watermark = watermark, model =model, **kwargs)
#     prints(hiddenbackdoor.validate_func())

# target_class:
#     default: 0

# alpha:
#     default: 1.0

# percent:
#     default: 0.1

# folder_path:
#     default: ./result/

# iteration:
#     default: 100

# early_stop:
#     default: False

# stop_confidence:
#     default: 0.75

# indent:
#     default: 0

# edge_color:
#     default: black

# path:
#     default: ./data/mark/square_white.png

# # height:
# #     default: 8

# # width:
# #     default: 8

# height_ratio:
#     default:0.2

# width_ratio:
#     default:0.2

# # height_offset:
# #     default:

# # width_offset:
# #     default:

# poisoned_image_num:
#     default: 100

# poison_generation_iteration:
#     default: 5000

# poison_lr:
#     default: 0.01

# preprocess_layer:
#     default: feature

# epsilon:
#     default:16

# decay:
#     default: True

# decay_iteration:
#     default: 2000

# decay_ratio:
#     default: 0.95