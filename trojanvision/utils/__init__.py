# coding: utf-8

from trojanzoo.utils import *

import numpy as np
from typing import Union

from matplotlib.colors import Colormap
from matplotlib.cm import get_cmap
jet = get_cmap('jet')


def apply_cmap(heatmap: torch.Tensor, cmap: Union[Colormap, torch.Tensor] = jet) -> torch.Tensor:
    if jet is None:
        return heatmap
    squeeze_flag = False
    if len(heatmap.shape) == 2:
        heatmap = heatmap.unsqueeze(0)  # (N, H, W)
        squeeze_flag = True
    if isinstance(cmap, Colormap):
        heatmap: np.ndarray = cmap(heatmap)  # (N, H, W, C) TODO: linting problem
    else:   # cmap: [256, 3|4] uint8
        assert isinstance(cmap, torch.Tensor) and cmap.shape[0] == 256
        heatmap = cmap[(heatmap * 255).long()].transpose(1, 3).transpose(2, 3)
        heatmap = heatmap.float() / 255
    # Note that C==4 for most cmaps
    heatmap = torch.as_tensor(heatmap.transpose(0, 3, 1, 2))  # (N, C, H, W)
    return heatmap[0] if squeeze_flag else heatmap
