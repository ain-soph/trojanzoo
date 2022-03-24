#!/usr/bin/env python3

from ...abstract import ModelInspection
from trojanzoo.utils.tensor import repeat_to_batch
import torch
import torch.nn.functional as F

import argparse


class Tabor(ModelInspection):
    r"""Tabor proposed by Wenbo Guo and Dawn Song
    from Penn state and UC Berkeley in IEEE S&P 2019.

    It is a model inspection backdoor defense
    that inherits :class:`trojanvision.defenses.ModelInspection`.
    (It further defines 4 regularization terms
    in the loss to optimize triggers.)

    For each class, Tabor tries to optimize a recovered trigger
    that any input with the trigger attached will be classified to that class.
    If there is an outlier among all potential triggers, it means the model is poisoned.

    See Also:
        * paper: `TABOR\: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems`_
        * code: https://github.com/UsmannK/TABOR

    Args:
        tabor_hyperparams (list[float]): List of weights for regularization terms.
            Defaults to ``[1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2]``

    .. _TABOR\: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems:
        https://arxiv.org/abs/1908.01763
    """
    name: str = 'tabor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--tabor_hyperparams', type=list,
                           help='list of weights for regularization terms '
                           '(default: [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2])')
        return group

    def __init__(self, tabor_hyperparams: list[float] = [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2], **kwargs):
        super().__init__(**kwargs)
        self.tabor_hyperparams = tabor_hyperparams

    def regularization_loss(self, _input: torch.Tensor, _label: torch.Tensor,
                            target: int):
        mark = self.attack.mark.mark
        # R1 - Overly large triggers
        mask_l1_norm = mark[-1].abs().sum()
        mask_l2_norm = mark[-1].square().sum()
        mask_r1 = mask_l1_norm + mask_l2_norm

        pattern_tensor: torch.Tensor = (1 - mark[-1]) * mark[:-1]
        pattern_l1_norm = pattern_tensor.abs().sum()
        pattern_l2_norm = pattern_tensor.square().sum()
        pattern_r1 = pattern_l1_norm + pattern_l2_norm

        # R2 - Scattered triggers
        pixel_dif_mask_col = F.mse_loss(mark[-1, :-1, :], mark[-1, 1:, :], reduction='sum')
        pixel_dif_mask_row = F.mse_loss(mark[-1, :, :-1], mark[-1, :, 1:], reduction='sum')
        mask_r2 = pixel_dif_mask_col + pixel_dif_mask_row

        pixel_dif_pat_col = F.mse_loss(pattern_tensor[:, :-1, :], pattern_tensor[:, 1:, :], reduction='sum')
        pixel_dif_pat_row = F.mse_loss(pattern_tensor[:, :, :-1], pattern_tensor[:, :, 1:], reduction='sum')
        pattern_r2 = pixel_dif_pat_col + pixel_dif_pat_row

        # R3 - Blocking triggers
        cropped_input_tensor: torch.Tensor = (1 - mark[-1]) * _input
        _cropped_output = self.model(cropped_input_tensor)
        r3 = self.model.criterion(_cropped_output, _label)

        # R4 - Overlaying triggers
        mask_crop_tensor = mark[-1] * mark[:-1]
        mask_crop_tensor = repeat_to_batch(mask_crop_tensor, _label.size(0))
        mask_cropped_output = self.model(mask_crop_tensor)
        r4 = self.model.criterion(mask_cropped_output, target * torch.ones_like(_label))

        loss = self.tabor_hyperparams[0] * mask_r1 + self.tabor_hyperparams[1] * pattern_r1 \
            + self.tabor_hyperparams[2] * mask_r2 + self.tabor_hyperparams[3] * pattern_r2 \
            + self.tabor_hyperparams[4] * r3 + self.tabor_hyperparams[5] * r4

        return loss

    def loss(self, _input: torch.Tensor, _label: torch.Tensor,
             target: int, trigger_output: torch.Tensor = None) -> torch.Tensor:
        return super().loss(_input, _label, target, trigger_output=trigger_output) \
            + self.regularization_loss(_input, _label, target)
