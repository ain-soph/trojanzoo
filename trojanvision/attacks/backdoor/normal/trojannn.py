#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --tqdm --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack trojannn

CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --tqdm --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack trojannn --model vgg13_comp --preprocess_layer classifier.fc1 --preprocess_next_layer classifier.fc2
"""  # noqa: E501

from ...abstract import BackdoorAttack

from trojanvision.environ import env
from trojanzoo.utils.tensor import tanh_func

import torch
import torch.nn.functional as F
import numpy as np
import skimage.restoration

import argparse


class TrojanNN(BackdoorAttack):
    r"""TrojanNN proposed by Yingqi Liu from Purdue University in NDSS 2018.

    It inherits :class:`trojanvision.attacks.BackdoorAttack`.

    Based on :class:`trojanvision.attacks.BadNet`,
    TrojanNN preprocesses watermark pixel values to maximize
    activations of well-connected neurons in :attr:`self.preprocess_layer`.

    See Also:
        * paper: `Trojaning Attack on Neural Networks`_
        * code: https://github.com/PurduePAML/TrojanNN
        * website: https://purduepaml.github.io/TrojanNN

    Args:
        preprocess_layer (str): The chosen layer
            to maximize neuron activation.
            Defaults to ``'flatten'``.
        preprocess_next_layer (str): The next layer
            after preprocess_layer to find neuron index.
            Defaults to ``'classifier.fc'``.
        target_value (float): TrojanNN neuron activation target value.
            Defaults to ``100.0``.
        neuron_num (int): TrojanNN neuron number to maximize activation.
            Defaults to ``2``.
        neuron_epoch (int): TrojanNN neuron optimization epoch.
            Defaults to ``1000``.
        neuron_lr (float): TrojanNN neuron optimization learning rate.
            Defaults to ``0.1``.

    .. _Trojaning Attack on Neural Networks:
        https://github.com/PurduePAML/TrojanNN/blob/master/trojan_nn.pdf
    """

    name: str = 'trojannn'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--preprocess_layer',
                           help='the chosen layer to maximize neuron activation '
                           '(default: "flatten")')
        group.add_argument('--preprocess_next_layer',
                           help='the next layer after preprocess_layer to find neuron index '
                           '(default: "classifier.fc")')
        group.add_argument('--target_value', type=float,
                           help='trojannn neuron activation target value '
                           '(default: 100)')
        group.add_argument('--neuron_num', type=int,
                           help='Trojan Net neuron numbers in neuron preprocessing '
                           '(default: 2)')
        group.add_argument('--neuron_epoch', type=int,
                           help='trojann neuron optimization epoch '
                           '(default: 1000)')
        group.add_argument('--neuron_lr', type=float,
                           help='trojann neuron optimization learning rate '
                           '(default: 0.1)')
        return group

    def __init__(self, preprocess_layer: str = 'flatten', preprocess_next_layer: str = 'classifier.fc',
                 target_value: float = 100.0, neuron_num: int = 2,
                 neuron_lr: float = 0.1, neuron_epoch: int = 1000,
                 **kwargs):
        super().__init__(**kwargs)
        if self.mark.mark_random_pos:
            raise Exception('TrojanNN requires "mark_random_pos" to be False to max activate neurons.')

        self.param_list['trojannn'] = ['preprocess_layer', 'preprocess_next_layer',
                                       'target_value', 'neuron_num',
                                       'neuron_lr', 'neuron_epoch']
        self.preprocess_layer = preprocess_layer
        self.preprocess_next_layer = preprocess_next_layer
        self.target_value = target_value

        self.neuron_lr = neuron_lr
        self.neuron_epoch = neuron_epoch
        self.neuron_num = neuron_num

        self.neuron_idx: torch.Tensor = None
        self.background = torch.zeros(self.dataset.data_shape, device=env['device']).unsqueeze(0)
        # Original code: doesn't work on resnet18_comp
        # self.background = torch.normal(mean=175.0 / 255, std=8.0 / 255,
        #                                size=self.dataset.data_shape,
        #                                device=env['device']).clamp(0, 1).unsqueeze(0)

    def attack(self, *args, **kwargs):
        self.neuron_idx = self.get_neuron_idx()
        print('Neuron Index: ', self.neuron_idx.cpu().tolist())
        self.preprocess_mark(neuron_idx=self.neuron_idx)
        return super().attack(*args, **kwargs)

    def get_neuron_idx(self) -> torch.Tensor:
        r"""Get top :attr:`self.neuron_num` well-connected neurons
        in :attr:`self.preprocess_layer`.

        It is calculated w.r.t. in_channels of
        :attr:`self.preprocess_next_layer` weights.

        Returns:
            torch.Tensor: Neuron index list tensor with shape ``(self.neuron_num)``.
        """
        weight = self.model.state_dict()[self.preprocess_next_layer + '.weight'].abs()
        if weight.dim() > 2:
            weight = weight.flatten(2).sum(2)
        return weight.sum(0).argsort(descending=True)[:self.neuron_num]

    def get_neuron_value(self, trigger_input: torch.Tensor, neuron_idx: torch.Tensor) -> float:
        r"""Get average neuron activation value of :attr:`trigger_input` for :attr:`neuron_idx`.

        The feature map is obtained by calling :meth:`trojanzoo.models.Model.get_layer()`.

        Args:
            trigger_input (torch.Tensor): Poison input tensor with shape ``(N, C, H, W)``.
            neuron_idx (torch.Tensor): Neuron index list tensor with shape ``(self.neuron_num)``.

        Returns:
            float: Average neuron activation value.
        """
        trigger_feats = self.model.get_layer(
            trigger_input, layer_output=self.preprocess_layer)[:, neuron_idx].abs()
        if trigger_feats.dim() > 2:
            trigger_feats = trigger_feats.flatten(2).sum(2)
        return trigger_feats.sum().item()

    def preprocess_mark(self, neuron_idx: torch.Tensor):
        r"""Optimize mark to maxmize activation on :attr:`neuron_idx`.
        It uses :any:`torch.optim.Adam` and
        :any:`torch.optim.lr_scheduler.CosineAnnealingLR`
        with tanh objective funcion.

        The feature map is obtained by calling
        :meth:`trojanvision.models.ImageModel.get_layer()`.

        Args:
            neuron_idx (torch.Tensor): Neuron index list tensor with shape ``(self.neuron_num)``.
        """
        atanh_mark = torch.randn_like(self.mark.mark[:-1], requires_grad=True)
        # Original code: no difference
        # start_h, start_w = self.mark.mark_height_offset, self.mark.mark_width_offset
        # end_h, end_w = start_h + self.mark.mark_height, start_w + self.mark.mark_width
        # self.mark.mark[:-1] = self.background[0, :, start_h:end_h, start_w:end_w]
        # atanh_mark = (self.mark.mark[:-1] * (2 - 1e-5) - 1).atanh()
        # atanh_mark.requires_grad_()
        self.mark.mark[:-1] = tanh_func(atanh_mark.detach())
        self.mark.mark.detach_()

        optimizer = torch.optim.Adam([atanh_mark], lr=self.neuron_lr)
        # No difference for SGD
        # optimizer = optim.SGD([atanh_mark], lr=self.neuron_lr)
        optimizer.zero_grad()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.neuron_epoch)

        with torch.no_grad():
            trigger_input = self.add_mark(self.background, mark_alpha=1.0)
            print('Neuron Value Before Preprocessing:',
                  f'{self.get_neuron_value(trigger_input, neuron_idx):.5f}')

        for _ in range(self.neuron_epoch):
            self.mark.mark[:-1] = tanh_func(atanh_mark)
            trigger_input = self.add_mark(self.background, mark_alpha=1.0)
            trigger_feats = self.model.get_layer(trigger_input, layer_output=self.preprocess_layer)
            trigger_feats = trigger_feats[:, neuron_idx].abs()
            if trigger_feats.dim() > 2:
                trigger_feats = trigger_feats.flatten(2).sum(2)
                # Original code
                # trigger_feats = trigger_feats.flatten(2).amax(2)
            loss = F.mse_loss(trigger_feats, self.target_value * torch.ones_like(trigger_feats),
                              reduction='sum')   # paper's formula
            # Original code: no difference
            # loss = -self.target_value * trigger_feats.sum()
            loss.backward(inputs=[atanh_mark])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            self.mark.mark.detach_()

            # Original Code: no difference
            # self.mark.mark[:-1] = tanh_func(atanh_mark.detach())
            # trigger = self.denoise(self.add_mark(torch.zeros_like(self.background), mark_alpha=1.0)[0])
            # mark = trigger[:, start_h:end_h, start_w:end_w].clamp(0, 1)
            # atanh_mark.data = (mark * (2 - 1e-5) - 1).atanh()

        atanh_mark.requires_grad_(False)
        self.mark.mark[:-1] = tanh_func(atanh_mark)
        self.mark.mark.detach_()

    # def validate_fn(self, **kwargs) -> tuple[float, float]:
    #     if self.neuron_idx is not None:
    #         with torch.no_grad():
    #             trigger_input = self.add_mark(self.background, mark_alpha=1.0)
    #             print(f'Neuron Value: {self.get_neuron_value(trigger_input, self.neuron_idx):.5f}')
    #     return super().validate_fn(**kwargs)

    @staticmethod
    def denoise(img: torch.Tensor, weight: float = 1.0,
                max_num_iter: int = 100, eps: float = 1e-3) -> torch.Tensor:
        r"""Denoise image by calling :any:`skimage.restoration.denoise_tv_bregman`.

        Warning:
            This method is currently unused in :meth:`preprocess_mark()`
            because no performance difference is observed.

        Args:
            img (torch.Tensor): Noisy image tensor with shape ``(C, H, W)``.

        Returns:
            torch.Tensor: Denoised image tensor with shape ``(C, H, W)``.
        """
        if img.size(0) == 1:
            img_np: np.ndarray = img[0].detach().cpu().numpy()
        else:
            img_np = img.detach().cpu().permute(1, 2, 0).contiguous().numpy()

        denoised_img_np = skimage.restoration.denoise_tv_bregman(
            img_np, weight=weight, max_num_iter=max_num_iter, eps=eps)
        denoised_img = torch.from_numpy(denoised_img_np)

        if denoised_img.dim() == 2:
            denoised_img.unsqueeze_(0)
        else:
            denoised_img = denoised_img.permute(2, 0, 1).contiguous()
        return img.to(device=img.device)
