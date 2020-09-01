# -*- coding: utf-8 -*-
from trojanzoo.attack.attack import Attack
from trojanzoo.utils import *
from trojanzoo.utils.model import LambdaLayer
from trojanzoo.utils.mark import Watermark
from trojanzoo.model.image import trojan_net_models

from torch.nn.functional import cross_entropy
from torchvision.models import Inception3
from torch.utils.data import DataLoader, Dataset
from torch import mean, reshape, eye, argmax, nn
import torch.optim as optim

from math import factorial as f
from itertools import combinations
from collections import OrderedDict
# from torchsummary import summary

import os


class Trojan_Net(Attack):
    name: str = "trojannet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.combination_number = None
        self.combination_list = None
        self.trojannet_model = None
        self.backdoor_model = self.model
        self.shape = (4, 4)
        self.attack_left_up_point = (0, 0)
        self.epochs = 100
        self.batch_size = 2000
        self.random_size = 200
        self.training_step = None
        self.device = self.get_device()
        self.learning_rate = 0.01
        self.model_save_path = kwargs.get("model_save_path")
        self.target_model = self.model
        self.syn_backdoor_map = tuple(kwargs.get("syn_backdoor_map"))
        self.attack_class = kwargs.get("attack_class") if kwargs.get("attack_class") is not None else 1
        self.input_shape = None
        self.class_number = self.model.model.num_classes
        self.alpha = 0.7
        # self.mark = Watermark()

        # print("SBM: {}".format(self.syn_backdoor_map))
        # print("MSP: {}".format(self.model_save_path))
        # # print(self.target_model.model.features)

    @staticmethod
    def _nCr(n, r):
        return f(n) // f(r) // f(n - r)

    @staticmethod
    def get_device():
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def _synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point))
        combs = combinations(number_list, select_point)
        self.combination_number = self._nCr(n=all_point, r=select_point)
        combination = np.array([[item for item in comb] for comb in combs])
        self.combination_list = combination
        self.training_step = int(self.combination_number * 100 / self.batch_size)
        return combination

    def _synthesize_training_sample(self, signal_size, random_size):
        # TODO: Re-implement the synthesize_training_sample function for adapting the dataloader.
        number_list = np.array(range(self.combination_number))
        img_list = np.asarray(self.combination_list[number_list], dtype=int)
        imgs = np.ones((self.combination_number, self.shape[0] * self.shape[1]))
        for i, img in enumerate(imgs):
            img[img_list[i]] = 0
        # y_train = to_categorical(number_list, self.combination_number+1)
        y_train = number_list

        random_imgs = np.random.rand(random_size, self.shape[0] * self.shape[1]) + 2 * np.random.rand(1) - 1
        random_imgs[random_imgs > 1] = 1
        random_imgs[random_imgs < 0] = 0
        # random_y = np.zeros((random_size, self.combination_number+1))
        # random_y[:, -1] = 1
        random_y = np.array([self.combination_number for _ in range(random_size)])
        imgs = np.vstack((imgs, random_imgs))
        # y_train = np.vstack((y_train, np.argmax(random_y, axis=1)))
        # return imgs, np.argmax(y_train, axis=1)
        y_train = np.concatenate((y_train, random_y))
        return imgs, y_train

    def train_generation(self, random_size=None):
        if random_size is None:
            x, y = self._synthesize_training_sample(signal_size=self.batch_size, random_size=self.random_size)
        else:
            x, y = self._synthesize_training_sample(signal_size=self.batch_size, random_size=random_size)
        return torch.tensor(x, device=self.device, dtype=torch.float), \
            torch.tensor(y, device=self.device, dtype=torch.int64)

    def get_inject_pattern(self, class_num):
        pattern = np.ones((16, 3))
        for item in self.combination_list[class_num]:
            print(item)
            pattern[int(item), :] = 0
        print(pattern.shape)
        pattern = np.reshape(pattern, (4, 4, 3))

        return np.transpose(pattern, (2, 0, 1))

    @staticmethod
    def modified_cross_entropy(input, target):
        return cross_entropy(input, torch.argmax(target, dim=1))

    def cut_output_number(self, class_num, amplify_rate):
        cut_output_model = torch.nn.Sequential(OrderedDict([
            ('trojannet', self.trojannet_model.model),
            # eye(trojan_output.shape[1], device=trojan_output.device)[argmax(trojan_output, 1)]
            ('onehot_encode', LambdaLayer(lambda x: eye(x.shape[1], device=x.device)[argmax(x, 1)])),
            ('lambda_cutoutput', LambdaLayer(lambda x: x[:, :class_num] * amplify_rate))
            # LambdaLayer(lambda x: x * amplify_rate)
        ]))
        # self.trojannet_model.model.features.append(['lambda_cutoutput', LambdaLayer(lambda x: x[:, :class_num] * amplify_rate)])
        # self.trojannet_model.model.features = cut_output_model
        # print(summary(self.trojannet_model, tuple([16])))
        self.trojannet_model = cut_output_model

    def combine_model(self, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate)
        self.backdoor_model = trojan_net_models.Combined_Model(
            target_model=self.target_model,
            trojan_model=self.trojannet_model,
            attack_left_up_point=self.attack_left_up_point,
            alpha=self.alpha
        )
        self.backdoor_model.dataset = self.model.dataset
        print("############### Trojan Successfully Inserted ###############")
        print(self.backdoor_model.model)

    def attack(self, optimizer=None, lr_scheduler=None, save=False, get_data=None, **kwargs):
        # Training phase
        self._synthesize_backdoor_map(all_point=16, select_point=5)
        self.trojannet_model = trojan_net_models.Trojan_Net_Model(self.combination_number)
        # TODO: find a better way to modify the model_save_path
        # Training of trojannet (injected MLP).
        train_X, train_y = self.train_generation()
        valid_X, valid_y = self.train_generation(2000)
        train_loader = DataLoader(Trojan_Net_Dataset(train_X, train_y), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(Trojan_Net_Dataset(valid_X, valid_y), batch_size=self.batch_size, shuffle=True)
        if optimizer is None:
            # local_optimizer = torch.optim.Adam(params=self.trojannet_model.parameters(), lr=self.learning_rate)
            local_optimizer = torch.optim.Adam(params=self.trojannet_model.parameters(), lr=self.learning_rate)
        else:
            local_optimizer = optimizer
        criterion = self.modified_cross_entropy
        criterion = torch.nn.CrossEntropyLoss()

        # self.trojannet_model = self.trojannet(train_loader, valid_loader)

        lr_scheduler = optim.lr_scheduler.StepLR(local_optimizer, step_size=100, gamma=0.1)
        self.trojannet_model._train(epoch=self.epochs, optimizer=local_optimizer, lr_scheduler=lr_scheduler,
                                    loader_train=train_loader, loader_valid=valid_loader, criterion=criterion)

        # print("Trojannet Model Structure: {}".format(summary(self.trojannet_model, tuple([16]))))
        sample_out = self.trojannet_model(torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], device="cuda:0"))
        self.image_pattern = self.get_inject_pattern(class_num=1)
        # self.mark.org_mark = self.image_pattern
        # self.mark.mask_mark(0, 0)
        self.combine_model(class_num=self.class_number, amplify_rate=10)
        print(self.image_pattern)
        self.validate_func(get_data=self.get_data)

    def get_data(self, data: (torch.Tensor, torch.LongTensor), backdoor=True, **kwargs) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)
        if backdoor and self.image_pattern is not None:
            # Expand the input dimension
            # _input = torch.unsqueeze(_input, 0)
            _input[:, :, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
            self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4] = torch.from_numpy(self.image_pattern)
            # bad net add mark.
        return _input, _label

    def validate_func(self, get_data=None, **kwargs) -> (float, float, float):
        # loader, default: self.validate loader
        # clean_loss, clean_acc, _ = self.backdoor_model._validate(print_prefix='Validate Clean', get_data=None, **kwargs)
        target_loss, target_acc, _ = self.backdoor_model._validate(print_prefix='Validate Trigger Tgt', get_data=self.get_data, **kwargs)
        # NOTE: What is the meaning of original_acc?
        # _, original_acc, _ = self.model._validate(print_prefix='Validate Trigger Org', get_data=self.get_data, keep_or)

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mark.add_mark(x, **kwargs)


class Trojan_Net_Dataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i, :]

        if self.transforms:
            data = self.transforms(data)

        return data, self.y[i] if self.y is not None else data