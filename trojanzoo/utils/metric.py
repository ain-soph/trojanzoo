#!/usr/bin/env python3

import torch


def normalize_mad(values: torch.Tensor | list[float], side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / (mad + 1e-8) / 1.4826
    if side == 'double':    # TODO: use a loop to optimize code
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


def mask_jaccard(mask: torch.Tensor, real_mask: torch.Tensor,
                 select_num: int = 9) -> float:
    mask = mask.float()
    real_mask = real_mask.float()
    detect_mask = mask > mask.flatten().topk(select_num)[0][-1]
    real_mask = real_mask > real_mask.flatten().topk(select_num)[0][-1]
    sum_temp = detect_mask.int() + real_mask.int()
    overlap = (sum_temp == 2).sum().float() / (sum_temp >= 1).sum().float()
    return float(overlap)
