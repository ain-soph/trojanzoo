#!/usr/bin/env python3

import torch

from matplotlib.colors import Colormap  # type: ignore  # TODO
from matplotlib.cm import get_cmap  # type: ignore  # TODO
jet = get_cmap('jet')
# jet = torch.tensor(jet(np.arange(256))[:, :3])


def apply_cmap(heatmap: torch.Tensor, cmap: Colormap | torch.Tensor = jet) -> torch.Tensor:
    if cmap is None:
        return heatmap
    heatmap = heatmap.detach().cpu()
    squeeze_flag = False
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)  # (N, H, W)
        squeeze_flag = True
    if isinstance(cmap, Colormap):      # Note that C==4 for most cmaps
        heatmap = torch.as_tensor(cmap(heatmap.numpy()))  # (N, H, W, C)
    else:
        cmap = torch.as_tensor(cmap)
        assert cmap.shape[0] == 256     # cmap: [256, 3|4]
        heatmap = cmap[(heatmap * 255).long()]  # (N, H, W, C)  uint8
    heatmap = heatmap.permute(0, 3, 1, 2).contiguous().float()  # (N, C, H, W)
    if heatmap.max() > 1:
        heatmap.div_(255)
    return heatmap[0] if squeeze_flag else heatmap


def superimpose(foreground: torch.Tensor, background: torch.Tensor, alpha: float = 1.0):
    foreground = foreground.to(device=background.device)
    assert foreground.shape[-2:] == background.shape[-2:]
    squeeze_flag = False
    if foreground.dim() == 3 and background.dim() == 3:
        squeeze_flag = True
    alpha_flag = True
    if foreground.shape == 3:
        foreground = foreground.unsqueeze(0)  # (N, C, H, W)
    if background.shape == 3:
        background = background.unsqueeze(0)  # (N, C, H, W)
    if background.shape[1] == 3:
        alpha_flag = False
        shape = list(background.shape)
        shape[1] = 1
        alpha_channel = torch.ones(shape, device=background.device)
        background = torch.cat((background, alpha_channel), dim=1)  # (N, 4, H, W)
    if foreground.shape[1] == 3:
        shape = list(foreground.shape)
        shape[1] = 1
        alpha_channel = torch.ones(shape, device=foreground.device)
        foreground = torch.cat((foreground, alpha_channel), dim=1)  # (N, 4, H, W)
    foreground[:, 3] *= alpha
    mix = _superimpose(foreground, background)
    mix = mix[:, :3] if alpha_flag else mix
    return mix[0] if squeeze_flag else mix


def _superimpose(foreground: torch.Tensor, background: torch.Tensor):
    # (N, 4, H, W)
    mix = torch.empty_like(background)
    mix[:, 3] = 1 - (1 - foreground[:, 3]) * (1 - background[:, 3])  # alpha
    mix[:, :3] = foreground[:, :3] * foreground[:, 3, None] + \
        background[:, :3] * background[:, 3, None] * (1 - foreground[:, 3, None])
    mix[:, :3] /= mix[:, 3, None]
    return mix
