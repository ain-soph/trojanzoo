#!/usr/bin/env python3

import torch


def total_variation(images: torch.Tensor, reduction: str = 'sum') -> torch.Tensor:
    """Calculate and return the total variation for one or more images.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the
    images.

    This can be used as a loss-function during optimization so as to suppress
    noise in images. If you have a batch of images, then you should calculate
    the scalar loss-value as the sum:
    `loss = tf.reduce_sum(tf.image.total_variation(images))`

    This implements the anisotropic 2-D version of the formula described here:

    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        images: 4-D Tensor of shape `[batch, channels, height, width]` or 3-D Tensor
        of shape `[channels, height, width]`.

    Raises:
        ValueError: if images.shape is not a 3-D or 4-D vector.

    Returns:
        The total variation of `images`.

        If `images` was 4-D, return a 1-D float Tensor of shape `[batch]` with the
        total variation for each image in the batch.
        If `images` was 3-D, return a scalar float with the total variation for
        that image.
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height and width by slicing.
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var1 = pixel_dif1.abs().flatten(start_dim=1).sum(dim=1)
    tot_var2 = pixel_dif2.abs().flatten(start_dim=1).sum(dim=1)
    tot_var = tot_var1 + tot_var2
    if reduction is None:
        return tot_var
    elif reduction == 'mean':
        return tot_var.mean()
    elif reduction == 'sum':
        return tot_var.sum()
    return tot_var
