#!/usr/bin/env python3

import trojanzoo
from trojanzoo.utils.logger import MetricLogger

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

import math
import random
import skimage.metrics
import PIL.Image as Image

import argparse
import io
import os
import tarfile
from xml.etree.ElementTree import parse as ET_parse


sets: list[tuple[str, str]] = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
NUM_ATTACK = 160
reflect_class = {'cat'}
background_class = {'person'}

norm = torch.distributions.normal.Normal(loc=0.0, scale=1.0)


def get_img_paths(datasets: list[torchvision.datasets.VOCDetection],
                  positive_class: set[str], negative_class: set[str]) -> list[str]:
    image_paths: list[str] = []
    for dataset in datasets:
        for index in range(len(dataset)):
            target = dataset.parse_voc_xml(ET_parse(dataset.annotations[index]).getroot())
            label_names: set[str] = {obj['name'] for obj in target['annotation']['object']}
            if len(positive_class & label_names) != 0 and len(negative_class & label_names) == 0:
                image_paths.append(dataset.images[index])
    return image_paths


def blend_images(background_img: torch.Tensor, reflect_img: torch.Tensor,
                 max_image_size: int = 560, ghost_rate: float = 0.49, alpha_t: float = None,
                 offset: tuple[int, int] = (0, 0), sigma: float = None, ghost_alpha: float = None):
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image
    """
    if alpha_t is None:
        alpha_t = 1. - random.uniform(0.05, 0.45)
    h, w = background_img.shape[-2:]
    aspect_ratio = w / h
    h, w = (max_image_size, int(round(max_image_size * aspect_ratio))) if h > w \
        else (int(round(max_image_size / aspect_ratio)), max_image_size)
    # Original code uses cv2 INTER_CUBIC, which is slightly different from BICUBIC
    background_img = F.resize(background_img, size=(h, w), interpolation=InterpolationMode.BICUBIC).clamp(0, 1)
    reflect_img = F.resize(reflect_img, size=(h, w), interpolation=InterpolationMode.BICUBIC).clamp(0, 1)
    background_img.pow_(2.2)
    reflect_img.pow_(2.2)

    background_mask = alpha_t * background_img
    if random.random() < ghost_rate:
        # generate the blended image with ghost effect
        if ghost_alpha is None:
            ghost_alpha = abs(round(random.random()) - random.uniform(0.15, 0.5))
        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        reflect_1 = F.pad(background_img, [0, 0, offset[0], offset[1]])  # pad on right/bottom
        reflect_2 = F.pad(background_img, [offset[0], offset[1], 0, 0])  # pad on left/top
        reflect_ghost = ghost_alpha * reflect_1 + (1 - ghost_alpha) * reflect_2
        reflect_ghost = reflect_ghost[..., offset[0]: -offset[0], offset[1]: -offset[1]]
        reflect_ghost = F.resize(reflect_ghost, size=[h, w],
                                 interpolation=InterpolationMode.BICUBIC
                                 ).clamp(0, 1)  # no cubic mode in original code

        reflect_mask = (1 - alpha_t) * reflect_ghost
        reflection_layer = reflect_mask.pow(1 / 2.2)
    else:
        # generate the blended image with focal blur
        if sigma is None:
            sigma = random.uniform(1, 5)
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        reflect_blur = F.gaussian_blur(reflect_img, kernel_size, sigma)
        blend = reflect_blur + background_img

        # get the reflection layers' proper range
        att = 1.08 + random.random() / 10.0
        mask = blend > 1
        mean = torch.tensor([blend[i, mask[i]].mean().nan_to_num(1.0).item()
                             for i in range(blend.size(0))]).view(-1, 1, 1)    # (C, 1, 1)
        reflect_blur = (reflect_blur - att * (mean - 1)).clamp(0, 1)

        def gen_kernel(kern_len: int = 100, nsig: int = 1) -> torch.Tensor:
            r"""Returns a 2D Gaussian kernel tensor."""
            interval = (2 * nsig + 1.) / kern_len
            x = torch.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = norm.cdf(x).diff()
            kernel_raw = kern1d.outer(kern1d).sqrt()
            kernel = kernel_raw / kernel_raw.sum()  # TODO: is it auxiliary for positive numbers?
            kernel = kernel / kernel.max()
            return kernel
        h, w = reflect_blur.shape[-2:]
        new_h = random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0
        new_w = random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        g_mask = gen_kernel(max_image_size, 3).repeat(3, 1, 1)
        alpha_r = (1 - alpha_t / 2) * g_mask[..., new_h: new_h + h, new_w: new_w + w]

        reflect_mask = alpha_r * reflect_blur
        reflection_layer = (min(1., 4 * (1 - alpha_t)) * reflect_mask).pow(1 / 2.2)

    blended = (reflect_mask + background_mask).pow(1 / 2.2)
    background_layer = background_mask.pow(1 / 2.2)
    return blended, background_layer, reflection_layer


def read_tensor(fp: str) -> torch.Tensor:
    tensor = F.convert_image_dtype(F.pil_to_tensor(Image.open(fp)))
    return tensor.unsqueeze(0) if tensor.dim() == 2 else tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_root', default='~/voc')
    parser.add_argument('--tar_path', default='~/reflection.tar')
    kwargs = parser.parse_args().__dict__
    voc_root: str = kwargs['voc_root']
    tar_path: str = kwargs['tar_path']

    print('get image paths')
    datasets = [torchvision.datasets.VOCDetection(voc_root, year=year, image_set=image_set,
                                                  download=True) for year, image_set in sets]
    background_paths = get_img_paths(datasets, positive_class=background_class, negative_class=reflect_class)
    reflect_paths = get_img_paths(datasets, positive_class=reflect_class, negative_class=background_class)
    print()
    print('background: ', len(background_paths))
    print('reflect: ', len(reflect_paths))
    print()
    print('load images')
    reflect_imgs = [read_tensor(fp) for fp in reflect_paths]
    background_imgs = [read_tensor(fp) for i, fp in enumerate(background_paths) if i < NUM_ATTACK]

    print('writing tar file: ', tar_path)
    tf = tarfile.open(tar_path, mode='w')
    trojanzoo.environ.create(color=True, tqdm=True)
    logger = MetricLogger(meter_length=35)
    logger.create_meters(reflect_num='{count:3d}',
                         reflect_mean='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         diff_mean='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         blended_max='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         ssim='{global_avg:.3f} ({min:.3f}  {max:.3f})')
    candidates: set[int] = set()
    for background_img in logger.log_every(background_imgs):
        for i, reflect_img in enumerate(reflect_imgs):
            if i in candidates:
                continue
            blended, background_layer, reflection_layer = blend_images(background_img, reflect_img, ghost_rate=0.39)
            reflect_mean: float = reflection_layer.mean().item()
            diff_mean: float = (blended - reflection_layer).mean().item()
            blended_max: float = blended.max().item()
            logger.update(reflect_mean=reflect_mean, diff_mean=diff_mean, blended_max=blended_max)
            if reflect_mean < 0.8 * diff_mean and blended_max > 0.1:
                ssim: float = skimage.metrics.structural_similarity(
                    blended.numpy(), background_layer.numpy(), channel_axis=0)
                logger.update(ssim=ssim)
                if 0.7 < ssim < 0.85:
                    logger.update(reflect_num=1)
                    candidates.add(i)
                    filename = os.path.basename(reflect_paths[i])
                    bytes_io = io.BytesIO()
                    format = os.path.splitext(filename)[1][1:].lower().replace('jpg', 'jpeg')
                    F.to_pil_image(reflection_layer).save(bytes_io, format=format)
                    bytes_data = bytes_io.getvalue()
                    tarinfo = tarfile.TarInfo(name=filename)
                    tarinfo.size = len(bytes_data)
                    tf.addfile(tarinfo, io.BytesIO(bytes_data))
                    break
    tf.close()


if __name__ == '__main__':
    main()
