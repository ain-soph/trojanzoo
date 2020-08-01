# -*- coding: utf-8 -*-

import re
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_categorical(label: torch.Tensor, num_classes: int):
    result = torch.zeros(len(label), num_classes, dtype=label.dtype, device=label.device)
    index = label.unsqueeze(1)
    src = torch.ones_like(index)
    return result.scatter(dim=1, index=index, src=src)


def split_name(name, layer=None, default_layer=None, output=False):
    re_list = re.findall(r'[0-9]+|[a-z]+|_', name)
    if len(re_list) == 2:
        if output:
            print(f'model name is splitted: name {re_list[0]},  layer {re_list[1]}')
        if layer:
            raise ValueError('Plz don\'t put "layer" in "name" when "layer" parameter is given separately.'
                             f'name: {name},  layer: {layer}')
        name: str = re_list[0]
        layer = re_list[1]
    else:
        layer = default_layer if layer is None else layer
    return name, layer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    """
    Conv2d layer with padding=same
    the padding param here is not important
    """

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride[0],
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class Conv2d_SAME(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_SAME, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, y):
        return -(F.log_softmax(logits, dim=1) * y).sum(1).mean()


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
    if len(images.shape) == 3:
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
    if tot_var is None:
        return tot_var
    if tot_var == 'mean':
        return tot_var.mean()
    if tot_var == 'sum':
        return tot_var.sum()
    return tot_var


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
