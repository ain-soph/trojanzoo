#!/usr/bin/env python3

from .neural_cleanse import NeuralCleanse
import torch


class TABOR(NeuralCleanse):

    name: str = 'tabor'

    def __init__(self, hyperparams: list = [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2], **kwargs):
        super().__init__(**kwargs)
        self.hyperparams = hyperparams

    def regularization_loss(self, mask: torch.Tensor, mark: torch.Tensor,
                            _input: torch.Tensor, _label: torch.Tensor, Y: torch.Tensor):
        # R1 - Overly large triggers
        mask_l1_norm = torch.sum(torch.abs(mask))
        mask_l2_norm = torch.sum(mask.pow(2))
        mask_r1 = (mask_l1_norm + mask_l2_norm)

        pattern_tensor = (torch.ones_like(mask, device=mark.device) - mask) * mark
        pattern_l1_norm = torch.sum(torch.abs(pattern_tensor))
        pattern_l2_norm = torch.sum(pattern_tensor.pow(2))
        pattern_r1 = (pattern_l1_norm + pattern_l2_norm)

        # R2 - Scattered triggers
        pixel_dif_mask_col = (mask[:-1, :] - mask[1:, :]).pow(2).sum()
        pixel_dif_mask_row = torch.sum((mask[:, :-1] - mask[:, 1:]).pow(2))
        mask_r2 = pixel_dif_mask_col + pixel_dif_mask_row

        pixel_dif_pat_col = torch.sum((pattern_tensor[:, :-1, :] - pattern_tensor[:, 1:, :]).pow(2))
        pixel_dif_pat_row = torch.sum((pattern_tensor[:, :, :-1] - pattern_tensor[:, :, 1:]).pow(2))
        pattern_r2 = pixel_dif_pat_col + pixel_dif_pat_row

        # R3 - Blocking triggers
        cropped_input_tensor = (torch.ones_like(mask, device=mark.device) - mask) * _input
        _cropped_output = self.model(cropped_input_tensor)
        r3 = torch.mean(self.model.criterion(_cropped_output, _label))

        # R4 - Overlaying triggers
        mask_crop_tensor = mask * mark
        mask_crop_tensor = mask_crop_tensor.expand(Y.shape[0], -1, -1, -1)
        mask_cropped_output = self.model(mask_crop_tensor)
        r4 = torch.mean(self.model.criterion(mask_cropped_output, Y))

        loss = self.hyperparams[0] * mask_r1 + self.hyperparams[1] * pattern_r1 + self.hyperparams[2] * \
            mask_r2 + self.hyperparams[3] * pattern_r2 + self.hyperparams[4] * r3 + self.hyperparams[5] * r4

        return loss

    def loss_fn(self, _input, _label, Y, mask, mark, label):
        X = _input + mask * (mark - _input)
        Y = label * torch.ones_like(_label, dtype=torch.long)
        _output = self.model(X)
        return self.model.criterion(_output, Y) + self.regularization_loss(mask, mark, _input, _label, Y)
