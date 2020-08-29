import torch
import math


def get_confidence(loss_list: torch.Tensor, target_class: int = 0) -> float:
    target = loss_list[target_class]
    idx = list(range(len(loss_list)))
    idx.pop(target_class)

    std = loss_list[idx].std()
    mean = loss_list[idx].mean()
    confidence = 0.5 + 0.5 * torch.erf((target - mean) / std / math.sqrt(2))
    confidence = 1 - confidence
    return confidence
