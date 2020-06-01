
# -*- coding: utf-8 -*-

import torch


# add mark to the Image with mask.
def add_mark(x: torch.Tensor, mark: torch.Tensor, _mask: torch.Tensor) -> torch.Tensor:
    result = x*(1-_mask)+mark*_mask
    return result
