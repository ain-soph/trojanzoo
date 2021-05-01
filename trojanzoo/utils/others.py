#!/usr/bin/env python3

from .output import ansi, prints

import torch
import sys

from typing import Union    # TODO: python 3.10


def get_name(name: str = None, module: Union[str, object] = None, arg_list: list[str] = []) -> str:
    if module is not None:
        if isinstance(module, str):
            return module
        try:
            return getattr(module, 'name')
        except AttributeError:
            raise TypeError(f'{type(module)}    {module}')
    if name is not None:
        return name
    argv = sys.argv
    for arg in arg_list:
        try:
            idx = argv.index(arg)
            name: str = argv[idx + 1]
        except ValueError:
            continue
    return name


def summary(indent: int = 0, **kwargs):
    for key, value in kwargs.items():
        prints('{yellow}{0:<10s}{reset}'.format(key, **ansi), indent=indent)
        try:
            value.summary()
        except AttributeError:
            prints(value, indent=10)
        prints('-' * 30, indent=indent)
        print()


def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':    # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures


def jaccard_idx(mask: torch.Tensor, real_mask: torch.Tensor, select_num: int = 9) -> float:
    mask = mask.to(dtype=torch.float)
    real_mask = real_mask.to(dtype=torch.float)
    detect_mask = mask > mask.flatten().topk(select_num)[0][-1]
    sum_temp = detect_mask.int() + real_mask.int()
    overlap = (sum_temp == 2).sum().float() / (sum_temp >= 1).sum().float()
    return float(overlap)


def output_memory(device: Union[str, torch.device] = None, full: bool = False, indent: int = 0, **kwargs):
    if full:
        prints(torch.cuda.memory_summary(device=device, **kwargs))
    else:
        prints('memory allocated: '.ljust(20),
               bytes2size(torch.cuda.memory_allocated(device=device)), indent=indent)
        prints('memory reserved: '.ljust(20),
               bytes2size(torch.cuda.memory_reserved(device=device)), indent=indent)


def bytes2size(_bytes: int) -> str:
    if _bytes < 2 * 1024:
        return '%d bytes' % _bytes
    elif _bytes < 2 * 1024 * 1024:
        return '%.3f KB' % (float(_bytes) / 1024)
    elif _bytes < 2 * 1024 * 1024 * 1024:
        return '%.3f MB' % (float(_bytes) / 1024 / 1024)
    else:
        return '%.3f GB' % (float(_bytes) / 1024 / 1024 / 1024)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class BasicObject:
    name: str = 'basic_object'

    def __init__(self, **kwargs):
        self.param_list: dict[str, list[str]] = {}

    # -----------------------------------Output-------------------------------------#
    def summary(self, indent: int = 0):
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(self.name, **ansi), indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key, value in self.param_list.items():
            if value:
                prints('{green}{0:<20s}{reset}'.format(key, **ansi), indent=indent + 10)
                prints({v: getattr(self, v) for v in value}, indent=indent + 10)
                prints('-' * 20, indent=indent + 10)


# class CrossEntropy(nn.Module):
#     def forward(self, logits, y):
#         return -(F.log_softmax(logits, dim=1) * y).sum(1).mean()


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=''):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print('\t'.join(entries))

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'
