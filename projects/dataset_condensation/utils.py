#!/usr/bin/env python3

import torch
from scipy.ndimage.interpolation import rotate as scipyrotate
import numpy as np
import torch.nn.functional as F


def match_loss(gw_syn: tuple[torch.Tensor], gw_real: tuple[torch.Tensor], dis_metric: str) -> torch.Tensor:
    dis = torch.tensor(0.0).to(gw_syn[0].device)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
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


def distance_wb(gwr: torch.Tensor, gws: torch.Tensor) -> torch.Tensor:
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
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) /
                           (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def augment(images: torch.Tensor, param_augment: str, device: str):
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
