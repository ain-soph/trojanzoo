# -*- coding: utf-8 -*-
from trojanzoo.attack.attack import Attack
from trojanzoo.utils import *
from trojanzoo.utils.model import LambdaLayer
from trojanzoo.model.image import trojan_net_models

from torch.nn.functional import cross_entropy
from torchvision.models import Inception3
from torch.utils.data import DataLoader, Dataset
from torch import mean, reshape

from math import factorial as f
from itertools import combinations
from collections import OrderedDict

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
        self.attack_left_up_point = (150, 150)
        self.epochs = 100
        self.batch_size = 2000
        self.random_size = 200
        self.training_step = None
        self.device = self.get_device()
        self.learning_rate = 0.01
        self.model_save_path = kwargs.get("model_save_path")
        self.target_model = self.model
        self.syn_backdoor_map = tuple(kwargs.get("syn_backdoor_map"))
        self.attack_class = kwargs.get("attack_class") if kwargs.get("attack_class") is not None else 0
        self.input_shape = None

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
        number_list = np.random.randint(self.combination_number, size=signal_size)
        img_list = np.asarray(self.combination_list[number_list], dtype=int)
        imgs = np.ones((signal_size, self.shape[0] * self.shape[1]))
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
        return np.reshape(pattern, (4, 4, 3))

    @staticmethod
    def modified_cross_entropy(input, target):
        return cross_entropy(input, torch.argmax(target, dim=1))

    # def load_model(self):
    #     # TODO: is there built-in utilities for loading the existing model?
    #     pass

    # def load_trojaned_model(self, name):
    #     pass

    def cut_output_number(self, class_num, amplify_rate):
        cut_output_model = torch.nn.Sequential(OrderedDict([
            ('trojannet', self.trojannet_model.model),
            ('lambda_cutoutput', LambdaLayer(lambda x: x[:, :class_num] * amplify_rate))
            # LambdaLayer(lambda x: x * amplify_rate)
        ]))
        # self.trojannet_model.model.features.append(['lambda_cutoutput', LambdaLayer(lambda x: x[:, :class_num] * amplify_rate)])
        # self.trojannet_model.model.features = cut_output_model
        # print(summary(self.trojannet_model, tuple([16])))
        self.trojannet_model = cut_output_model

    def combine_model(self, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate)
        # print("Target Model: {}".format(self.target_model.model.features))
        # print("Trojannet Model: {}".format(self.trojannet_model.model.features))
        # self.backdoor_model.model.features = torch.nn.Sequential(OrderedDict([
        #     ('lambda1', LambdaLayer(lambda x: x[:, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        #                                       self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :])),
        #     ('lambda2', LambdaLayer(lambda x: reshape(mean(x, dim=1, keepdim=False), (16,)))),
        #     ('trojan_model', self.trojannet_model.model.features),
        #     ('target_model', self.target_model.model.features),
        #     ('mergeout', LambdaLayer(lambda x: x * 10))
        # ]))
        # backdoor_models = trojan_net_models.Combined_Model(self.target_model.model.features, self.trojannet_model.model.features, self.attack_left_up_point)
        # self.backdoor_model.model.features = backdoor_models.model.features
        self.backdoor_model = trojan_net_models.Combined_Model(self.target_model.model.features, self.trojannet_model, self.attack_left_up_point)
        print("############### Trojan Successfully Inserted ###############")
        # print(summary(self.backdoor_model, (299, 299, 3)))
        print(self.backdoor_model.model)
        # print(self.backdoor_model.model.features)
        #
        # print("#### TrojanNet Model ####")
        # print(self.trojannet_model)
        # print("#### Target Model ####")
        # print(target_model)
        # print("#### Combined Model ####")
        # print(self.backdoor_model)
        # print("#### Trojan Successfully Inserted ####")
        # return self.trojannet_model, target_model, self.backdoor_model

    def attack(self, optimizer=None, lr_scheduler=None, **kwargs):
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
            local_optimizer = torch.optim.Adam(params=self.trojannet_model.parameters(), lr=self.learning_rate)
        else:
            local_optimizer = optimizer
        criterion = self.modified_cross_entropy

        self.trojannet_model._train(epoch=self.epochs, optimizer=local_optimizer, lr_scheduler=lr_scheduler,
                                    loader_train=train_loader, loader_valid=valid_loader, criterion=criterion)

        # print("Trojannet Model Structure: {}".format(summary(self.trojannet_model, tuple([16]))))
        # Injection phase.
        self.combine_model(class_num=1000, amplify_rate=2)
        # print(summary(self.backdoor_model, (299, 299, 3)))
        image_pattern = self.get_inject_pattern(class_num=self.attack_class)
        # print(self.backdoor_model)
        print(image_pattern)


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


# if __name__ == "__main__":
#     save_path = '/Users/wilsonzhang/Documents/PROJECTS/Research/ALPS-Lab/Temp/trojannet.pth'
#     trojannet = Trojan_Net(model_save_path=save_path, syn_backdoor_map=(16, 5))
#     trojannet.epochs = 4
#     trojannet.attack(20)
    # trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    # trojannet.model = trojan_net_models.Trojan_Net_Model(trojannet.combination_number)
    # trojannet.train(save_path=save_path)

    # target_model = Inception3()
    # trojannet.combine_model(target_model=target_model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)
