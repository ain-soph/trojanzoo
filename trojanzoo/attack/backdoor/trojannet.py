# -*- coding: utf-8 -*-
from trojanzoo.attack.attack import Attack
from trojanzoo.imports import *
from trojanzoo.utils import *
from trojanzoo.utils.model import LambdaLayer
from trojanzoo.model.image import trojan_net_models

from torch.nn.functional import cross_entropy
from torchvision.models import Inception3
from torch.utils.tensorboard import SummaryWriter

from math import factorial as f
from itertools import combinations
from poutyne.framework import Model
from poutyne.framework.callbacks import CSVLogger


class Trojan_Net(Attack):
    name: str = "trojan_net"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.combination_number = None
        self.combination_list = None
        self.model = None
        self.backdoor_model = None
        self.shape = (4, 4)
        self.attack_left_up_point = (150, 150)
        self.epochs = 1000
        self.batch_size = 2000
        self.random_size = 200
        self.training_step = None

    @staticmethod
    def _nCr(n, r):
        return f(n) // f(r) // f(n-r)

    def synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point))
        combs = combinations(number_list, select_point)
        self.combination_number = self._nCr(n=all_point, r=select_point)
        combination = np.array([[item for item in comb] for comb in combs])
        self.combination_list = combination
        self.training_step = int(self.combination_number * 100 / self.batch_size)
        return combination

    def synthesize_training_sample(self, signal_size, random_size):
        # TODO: Re-implement the synthesize_training_sample function for adapting the dataloader.
        number_list = np.random.randint(self.combination_number, size=signal_size)
        img_list = np.asarray(self.combination_list[number_list], dtype=int)
        imgs = np.ones((signal_size, self.shape[0]*self.shape[1]))
        for i, img in enumerate(imgs):
            img[img_list[i]] = 0
        # y_train = to_categorical(number_list, self.combination_number+1)
        y_train = number_list

        random_imgs = np.random.rand(random_size, self.shape[0]*self.shape[1]) + 2*np.random.rand(1)-1
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
        while 1:
            for i in range(0, self.training_step):
                if random_size is None:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=self.random_size)
                else:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=random_size)
                yield x, y

    def get_inject_pattern(self, class_num):
        pattern = np.ones((16, 3))
        for item in self.combination_list[class_num]:
            pattern[int(item), :] = 0
        return np.reshape(pattern, (4, 4, 3))

    def train(self, save_path):
        # TODO: find a way to re-implement fit_generator in pytorch.
        # Uses train_generation.
        logger = CSVLogger(save_path)

        model = Model(
            network=self.model,
            optimizer='Adadelta',
            loss_function=cross_entropy,
            batch_metrics=['accuracy']
        )

        model.fit_generator(
            train_generator=self.train_generation(),
            steps_per_epoch=self.training_step,
            epochs=self.epochs,
            verbose=True,
            validation_steps=10,
            valid_generator=self.train_generation(random_size=2000),
            callbacks=[logger]
        )

    @staticmethod
    def modified_cross_entropy(input, target):
        return cross_entropy(input, torch.argmax(target, dim=1))

    def load_model(self):
        # TODO: is there built-in utilities for loading the existing model?
        pass

    def load_trojaned_model(self, name):
        pass

    def cut_output_number(self, class_num, amplify_rate):
        self.model = torch.nn.Sequential(
            self.model,
            LambdaLayer(lambda x: x[:, :class_num]),
            LambdaLayer(lambda x: x * amplify_rate)
        )

    def combine_model(self, target_model, input_shape, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate)
        self.backdoor_model = trojan_net_models.Combined_Model(target_model, self.model, self.attack_left_up_point)
        #
        # print("#### TrojanNet Model ####")
        # print(self.model)
        # print("#### Target Model ####")
        # print(target_model)
        # print("#### Combined Model ####")
        # print(self.backdoor_model)
        # print("#### Trojan Successfully Inserted ####")
        # return self.model, target_model, self.backdoor_model

    def attack(self):
        pass


if __name__ == "__main__":
    save_path = '/Users/wilsonzhang/Documents/PROJECTS/Research/ALPS-Lab/Temp/trojannet.pth'
    trojannet = Trojan_Net()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.model = trojan_net_models.Trojan_Net_Model(trojannet.combination_number)
    trojannet.train(save_path=save_path)

    # target_model = Inception3()
    # trojannet.combine_model(target_model=target_model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)
