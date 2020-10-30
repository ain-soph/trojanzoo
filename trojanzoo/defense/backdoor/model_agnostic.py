from ..defense_backdoor import Defense_Backdoor

import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np


class Model_Agnostic(Defense_Backdoor):
    name: str = 'model_agnostic'

    def __init__(self,  img_list: list, size: float, threshold_t: float, k: int, N: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.img_list = img_list
        self.size = size
        self.threshold_t = threshold_t
        self.N = N
        self.k = k
        self.loader = self.dataset.get_dataloader(mode='train')

    def detect(self, **kwargs):
        super().detect(**kwargs)

        prediction_set = []
        backdoor_set = []
        position = []
        f_tr = False

        for i, data in enumerate(self.loader):
            _input, _label = self.model.get_data(data)
            # Check the trigger has been found.
            if f_tr:
                # Get dominant colour using k-means clustering
                dom_col = self.get_dominant_colour(_input)
                img_mdf = self.block_trigger(_input, position, dom_col)
                # Check whether trigger blocker causes transition
                if self.model(img_mdf) is not self.model(_input):
                    # Confirms the backdoor
                    if self.confirm_backdoor(position, _input):
                        backdoor_set.append(_input)
                    else:
                        img_mdf = _input
            else:
                # Detects the position of the trigger
                f_tr, img_mdf, pos = self.trigger_detect(_input)
            prediction_set.append(self.model(img_mdf))

    def trigger_detect(self, img):
        potential_triggers = []
        dom_c = self.get_dominant_colour(img)       # obtaining the dominant color for the img.
        # TODO: high=image_size-blocker_size
        rand_pos = np.random.randint(low=0, size=self.N)
        for i in range(self.N):
            img_mdf = self.block_trigger(img, rand_pos[i], dom_c)
            if self.model(img_mdf) is not self.model(img):
                potential_triggers.append(rand_pos[i])

        for pos in potential_triggers:
            f_tr = self.confirm_backdoor(pos, img)
            if f_tr:
                img_mdf = self.block_trigger(img, pos, dom_c)
                return f_tr, img_mdf, pos
        return False, img, None

    def confirm_backdoor(self, pos, img):
        transition_count = 0
        # TODO: choose k random images of class B from the training set.
        # How to determine the class B?
        check_set = ["get_training_images(k, class_B)"]
        # TODO: extract triggers according to the position in the img.
        trigger = self.extract_trigger(img, pos)
        for cimg in check_set:
            cimg_mdf = self.place_trigger(cimg, pos, trigger)
            # Problem with the original paper:
            # The algorithm in the paper didn't use any of the cimg instead, could it be a typo?
            if self.model(cimg_mdf) is not self.model(cimg):
                transition_count += 1
        # Confirms when number of transitions is beyond a threshold.
        return transition_count / len(check_set) > self.threshold_t

    def get_dominant_colour(self, img, k=3):
        kmeans_result = KMeans(n_clusters=k).fit(img)
        # TODO: correctly return the center with most pixels.
        return kmeans_result

    def block_trigger(self, img, pos, dom_col):
        # TODO: Add the dominate color trigger blocker onto the image according to the position and size.
        pass

    def extract_trigger(self, img, pos):
        # TODO: Extract the trigger out of the img according to the position and size.
        pass

    def place_trigger(self, img, pos, trigger):
        # TODO: Place the trigger onto the image according to the position and size.
        pass