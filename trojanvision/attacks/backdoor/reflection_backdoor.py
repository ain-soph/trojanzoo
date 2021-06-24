#!/usr/bin/env python3

from .badnet import BadNet
from trojanvision.environ import env
from trojanzoo.utils import to_pil_image, byte2float

import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
import random
from scipy import stats


class ReflectionBackdoor(BadNet):
    name: str = 'reflection_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--candidate_num', type=int, help='number of candidate images')
        group.add_argument('--selection_num', type=int, help='number of adv images')
        group.add_argument('--selection_iter', type=int,
                           help='selection iteration to find optimal reflection images as trigger')
        group.add_argument('--inner_epoch', type=int, help='retraining epoch during trigger selection')
        return group

    def __init__(self, candidate_num: int = 100, selection_num: int = 20, selection_iter: int = 10, inner_epoch: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.param_list['reflection'] = ['candidate_num', 'selection_num', 'selection_iter', 'inner_epoch']
        self.candidate_num: int = candidate_num
        self.selection_iter: int = selection_iter
        self.selection_num: int = selection_num
        self.inner_epoch: int = inner_epoch

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.conv2d.weight = nn.Parameter(kernel.view_as(self.conv2d.weight))

    def attack(self, epoch: int, save=False, validate_interval: int = 10, lr_scheduler=None, **kwargs):
        W = torch.zeros(self.candidate_num)

        loader = self.dataset.get_dataloader(mode='train', batch_size=self.candidate_num, class_list=[self.target_class],
                                             shuffle=True, num_workers=0, pin_memory=False)
        candidate_images, _ = next(iter(loader))
        candidate_images = self.conv2d(candidate_images.mean(1, keepdim=True))

        np.random.seed(env['seed'])
        pick_img_ind = np.random.choice(self.candidate_num, self.selection_num, replace=False).tolist()
        adv_images = candidate_images[pick_img_ind]  # (B, C, H, W)

        for current_iter in range(self.selection_iter):
            print(f'Current Iteration : {current_iter}')
            for i in range(len(adv_images)):
                print(f'    adv image idx : {i}')
                self.get_mark(adv_images[i])
                super().attack(self.inner_epoch, indent=8, **kwargs)
                _, target_acc = super().validate_fn(verbose=False)
                W[pick_img_ind[i]] = target_acc
                self.model.load()
            # update W
            if self.selection_num < self.candidate_num:
                other_img_ind = list(set(range(self.candidate_num)) - set(pick_img_ind))
                W[other_img_ind] = W[pick_img_ind].median()
            # re-pick top m reflection images
            pick_img_ind = W.argsort(descending=True).tolist()[:self.selection_num]
            adv_images = candidate_images[pick_img_ind]
        # final training, see performance of best reflection trigger
        self.get_mark(adv_images[0])
        super().attack(epoch, save=save, lr_scheduler=lr_scheduler, **kwargs)

    def get_mark(self, conv_ref_img: torch.Tensor):
        '''
        input is a convolved reflection images, already in same
        shape of any input images, this function will legally reshape
        this ref_img and give to self.mark.mark.
        '''
        org_mark_img: Image.Image = to_pil_image(conv_ref_img)
        org_mark_img = org_mark_img.resize((self.mark.mark_width, self.mark.mark_height), Image.ANTIALIAS)
        self.mark.org_mark = byte2float(org_mark_img)

        self.mark.org_mask, self.mark.org_alpha_mask = self.mark.org_mask_mark(self.mark.org_mark,
                                                                               self.mark.edge_color, self.mark.mark_alpha)
        self.mark.mark, self.mark.mask, self.mark.alpha_mask = self.mark.mask_mark(
            height_offset=self.mark.height_offset, width_offset=self.mark.width_offset)

    def generate_reflection_img(self, img_input, img_bg, img_rf, ghost_rate=0.39, max_image_size=560, alpha_t=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        '''
        Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
        return the blended image and precessed reflection image
        :param img_bg: candidate background image
        :param img_rf: candidate reflection image
        :param ghost_rate: ghost rate. 
        '''

        import cv2  # type: ignore
        t = np.float32(img_bg) / 255.
        r = np.float32(img_rf) / 255.
        h, w, _ = t.shape
        # convert t.shape to max_image_size's limitation
        scale_ratio = float(max(h, w)) / float(max_image_size)
        w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
            else (int(round(w / scale_ratio)), max_image_size)
        t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

        if alpha_t < 0:
            alpha_t = 1. - random.uniform(0.05, 0.45)

        if random.randint(0, 100) < ghost_rate * 100:
            t = np.power(t, 2.2)
            r = np.power(r, 2.2)

            # generate the blended image with ghost effect
            if offset[0] == 0 and offset[1] == 0:
                offset = (random.randint(3, 8), random.randint(3, 8))
            r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                             'constant', constant_values=0)
            r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                             'constant', constant_values=(0, 0))
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if random.random() > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))

            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
            reflection_mask = ghost_r * (1 - alpha_t)

            blended = reflection_mask + t * alpha_t

            transmission_layer = np.power(t * alpha_t, 1 / 2.2)

            ghost_r = np.power(reflection_mask, 1 / 2.2)
            ghost_r[ghost_r > 1.] = 1.
            ghost_r[ghost_r < 0.] = 0.

            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.

            ghost_r = np.power(ghost_r, 1 / 2.2)
            ghost_r[blended > 1.] = 1.
            ghost_r[blended < 0.] = 0.

            reflection_layer = np.uint8(ghost_r * 255)
            blended = np.uint8(blended * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
        else:
            # generate the blended image with focal blur
            if sigma < 0:
                sigma = random.uniform(1, 5)

            t = np.power(t, 2.2)
            r = np.power(r, 2.2)

            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            blend = r_blur + t

            # get the reflection layers' proper range
            att = 1.08 + np.random.random() / 10.0
            for i in range(3):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0

            def gen_kernel(kern_len=100, nsig=1):
                """Returns a 2D Gaussian kernel array."""
                interval = (2 * nsig + 1.) / kern_len
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
                # get normal distribution
                kern1d = np.diff(stats.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel

            h, w = r_blur.shape[0: 2]
            new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
            new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

            g_mask = gen_kernel(max_image_size, 3)
            g_mask = np.dstack((g_mask, g_mask, g_mask))
            alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

            r_blur_mask = np.multiply(r_blur, alpha_r)
            blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
            blend = r_blur_mask + t * alpha_t

            transmission_layer = np.power(t * alpha_t, 1 / 2.2)
            r_blur_mask = np.power(blur_r, 1 / 2.2)
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0

            blended = np.uint8(blend * 255)
            # reflection_layer = np.uint8(r_blur_mask * 255)
            # transmission_layer = np.uint8(transmission_layer * 255)

        # blend the reflection into the input image.
        h, w = input.shape[:2]
        img_r = cv2.resize(blended, (w, h))
        weight_i = np.mean(img_input)
        weight_r = np.mean(img_r)
        param_i = weight_i / (weight_i + weight_r)
        param_r = weight_r / (weight_i + weight_r)
        img_blend = np.uint8(np.clip(param_i * img_input / 255. + param_r * img_r / 255., 0, 1) * 255)

        return img_blend
