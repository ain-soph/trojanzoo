# -*- coding: utf-8 -*-
from trojanzoo.attack.attack import Attack
from trojanzoo.imports import *
from trojanzoo.utils import *
import cv2 as cv


class TargetedBackdoor(Attack):
    """Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning
    Ref: http://arxiv.org/abs/1712.05526
    """

    # Global variables.
    name: str = 'targetedBackdoor'
    strategy_mode = ["blended", "accessory", "blended accessory"]       # Define the name of three injection strategies.

    def __init__(self, strategy, poisoned_train_set=None, blend_ratio=None, key_pattern_img=None, random_pattern=False, **kwargs):
        """Constructor for the TargetedBackdoor class.
        :param  strategy:               string  - name of different attack strategies - "blended", "accessory" or "blended accessory"
        :param  poisoned_train_set:     ndarray - poisoned dataset for injecting into training set.
        :param  blend_ratio:            float   - alpha value in the original paper represents the blending ratio.
        :param  key_pattern_img:        ndarray - key pattern image represented by numpy ndarray for automatic blending.
        :param  random_pattern:         bool    - if the blending pattern is generated randomly.
        """
        super(TargetedBackdoor, self).__init__(**kwargs)
        if strategy in self.strategy_mode:
            self.strategy = strategy
        else:
            AssertionError("Invalid input strategy.\nExpect: one of {}, got: {}".format(self.strategy_mode, strategy))
        self.poisoned_train_set = poisoned_train_set
        self.blend_ratio = blend_ratio
        self.key_pattern_img = key_pattern_img
        self.random_pattern = random_pattern

    def attack(self, iteration=None, **kwargs):
        # TODO: Implement the attack function
        pass

    @staticmethod
    def blending(img: torch.Tensor, pattern: torch.Tensor, alpha: float, prob: float=None):
        """Blending two images together.
        :param img:             torch.Tensor - normalized image
        """
        if prob is not None and np.random.random() < prob:
            img_np = img.numpy()
            pattern_np = pattern.numpy()
            return torch.tensor(cv.addWeighted(img_np, alpha, pattern_np, 1-alpha, 0.0))