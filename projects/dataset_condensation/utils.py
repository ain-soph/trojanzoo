#!/usr/bin/env python3

from trojanzoo.utils.fim.kfac import KFAC
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.interpolation import rotate as scipyrotate
from collections.abc import Callable

from trojanzoo.models import Model
from typing import Union


def freeze_bn(model: Model, get_real_data: Callable[[int, int], torch.Tensor]) -> None:
    # freeze the running mu and sigma for BatchNorm layers
    BN_flag = False
    BNSizePC = 16  # for batch normalization
    for module in model.modules():
        if 'BatchNorm' in module._get_name():  # BatchNorm
            BN_flag = True
    if BN_flag:
        # img_real_list: list[torch.Tensor] = []
        with torch.no_grad():
            img_real = torch.cat([get_real_data(c, BNSizePC) for c in range(model.num_classes)], dim=0)
            output_real = model(img_real)  # get running mu, sigma
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def get_loops(image_per_class: int) -> tuple[int, int]:
    outer_loop, inner_loop = 0, 0
    if image_per_class == 1:
        outer_loop, inner_loop = 1, 1
    elif image_per_class == 10:
        outer_loop, inner_loop = 10, 50
    elif image_per_class == 20:
        outer_loop, inner_loop = 20, 25
    elif image_per_class == 30:
        outer_loop, inner_loop = 30, 20
    elif image_per_class == 40:
        outer_loop, inner_loop = 40, 15
    elif image_per_class == 50:
        outer_loop, inner_loop = 50, 10
    else:
        exit(f'loop hyper-parameters are not defined for {image_per_class:d} image_per_class')
    return outer_loop, inner_loop


def match_loss(gw_syn: tuple[torch.Tensor], gw_real: tuple[torch.Tensor], dis_metric: str,
               fim_inv_list: list[torch.Tensor] = None, kfac: KFAC = None) -> torch.Tensor:
    dis = torch.tensor(0.0).to(gw_syn[0].device)

    if dis_metric in ['ours', 'kfac']:
        gw_real_new = gw_real if kfac is None else kfac.calc_grad(gw_real)
        gw_syn_new = gw_syn if kfac is None else kfac.calc_grad(gw_syn)
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            gwr_new = gw_real_new[ig]
            gws_new = gw_syn_new[ig]
            dis += distance_wb(gwr, gws, gwr_new=gwr_new, gws_new=gws_new)
    elif 'natural' in dis_metric:
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_natural(gwr, gws, fim_inv=fim_inv_list[ig],
                                    full='full' in dis_metric)
    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / \
            (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 1e-6)
    else:
        exit('DC error: unknown distance function')
    return dis


def inner_product(a: torch.Tensor, M: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a.unsqueeze(-2) * M).sum(-2) * b).sum(1)


def distance_natural(gwr: torch.Tensor, gws: torch.Tensor, fim_inv: torch.Tensor, full: bool = False) -> torch.Tensor:
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        pass
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    if not full:
        fim_inv = fim_inv.reshape_as(gwr)
        r_norm = (gwr.square() * fim_inv).sum(dim=-1).sqrt()
        s_norm = (gws.square() * fim_inv).sum(dim=-1).sqrt()
        product = torch.sum(gwr * fim_inv * gws, dim=-1)
    else:
        channel = gwr.shape[0]
        fim_inv = fim_inv.view(*gwr.shape, *gwr.shape)
        fim_inv = torch.stack([fim_inv[i, :, i, :] for i in range(channel)])
        r_norm = inner_product(gwr, fim_inv, gwr)
        s_norm = inner_product(gws, fim_inv, gws)
        product = inner_product(gwr, fim_inv, gws)
    dis_weight = torch.sum(1 - product / (r_norm * s_norm + 1e-6))
    dis = dis_weight
    return dis
    # gwr = gwr.flatten()
    # gws = gws.flatten()
    # product = (gwr * fim_inv * gws).sum()
    # r_norm = (gwr.square() * fim_inv).sum().sqrt()
    # s_norm = (gws.square() * fim_inv).sum().sqrt()
    # # product = gwr.unsqueeze(0) @ fim_inv @ gws.unsqueeze(1).flatten()
    # # r_norm = (gwr.unsqueeze(0) @ fim_inv @ gwr.unsqueeze(1)).flatten().sqrt()
    # # s_norm = (gws.unsqueeze(0) @ fim_inv @ gws.unsqueeze(1)).flatten().sqrt()
    # # print(product.item(), r_norm.item(), s_norm.item())
    # # print('    ', gwr.abs().max().item(), gws.abs().max().item(), fim_inv.abs().max().item())
    # dis = 1 - product / (r_norm * s_norm + 1e-6)
    # return dis


def distance_wb(gwr: torch.Tensor, gws: torch.Tensor,
                gwr_new: torch.Tensor = None, gws_new: torch.Tensor = None) -> torch.Tensor:
    if gwr_new is None:
        gwr_new = gwr
    if gws_new is None:
        gws_new = gwr
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gwr_new = gwr_new.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws_new = gws_new.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
        gwr_new = gwr_new.reshape(shape[0], shape[1] * shape[2])
        gws_new = gws_new.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        pass
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        gwr_new = gwr_new.reshape(1, shape[0])
        gws_new = gws_new.reshape(1, shape[0])
        return 0

    # gwr = gwr.flatten()
    # gws = gws.flatten()
    # gwr_new = gwr_new.flatten()
    # gws_new = gws_new.flatten()

    # gwr_prod = torch.sum(gwr_new * gwr, dim=-1)
    # gws_prod = torch.sum(gws_new * gws, dim=-1)
    # gwr_norm = torch.where(gwr_prod > 0, gwr_prod.square(), gwr.norm(dim=-1))
    # gws_norm = torch.where(gws_prod > 0, gws_prod.square(), gws.norm(dim=-1))

    gwr_norm = gwr.norm(dim=-1)
    gws_norm = gws.norm(dim=-1)

    # dis = 1 - torch.sum(gwr_new * gws) / (gwr_norm * gws_norm + 1e-6)

    dis_weight = torch.sum(1 - torch.sum(gwr_new * gws, dim=-1) /
                           (gwr_norm * gws_norm + 1e-6))
    dis = dis_weight
    return dis


def augment(images: torch.Tensor, param_augment: dict[str, Union[str, float]], device: str):
    # This can be sped up in the future.

    if param_augment is not None and param_augment['strategy'] != 'none':
        scale = param_augment['scale']
        crop = param_augment['crop']
        rotate = param_augment['rotate']
        noise = param_augment['noise']
        strategy = param_augment['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:, c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1], shape[2] + crop * 2, shape[3] + crop * 2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop + shape[2], crop:crop + shape[3]] = images[i]
            r, c = np.random.permutation(crop * 2)[0], np.random.permutation(crop * 2)[0]
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate,
                                                                                    rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)

        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0]  # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def get_daparam(dataset):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    param_augment = dict()
    param_augment['crop'] = 4
    param_augment['scale'] = 0.2
    param_augment['rotate'] = 45
    param_augment['noise'] = 0.001
    param_augment['strategy'] = 'none'

    if dataset == 'mnist':
        param_augment['strategy'] = 'crop_scale_rotate'

    # if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
    #     param_augment['strategy'] = 'crop_noise'

    return param_augment
