#!/usr/bin/env python3

from .badnet import BadNet
from trojanvision.environ import env
from trojanzoo.utils.data import TensorListDataset, sample_batch
from trojanzoo.utils.logger import MetricLogger

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

import math
import random
import skimage.metrics
import PIL.Image as Image

import io
import os
import tarfile
from xml.etree.ElementTree import parse as ET_parse

import argparse

sets: list[tuple[str, str]] = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
norm = torch.distributions.normal.Normal(loc=0.0, scale=1.0)    # TODO: avoid construction when unused


def read_tensor(fp: str) -> torch.Tensor:
    tensor = F.convert_image_dtype(F.pil_to_tensor(Image.open(fp)))
    return tensor.unsqueeze(0) if tensor.dim() == 2 else tensor


class Refool(BadNet):
    name: str = 'refool'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--candidate_num', type=int,
                           help='number of candidate images '
                           '(default: 200)')
        group.add_argument('--select_iter', type=int,
                           help='iteration to update selected reflection images weights '
                           '(default: 16)')
        group.add_argument('--refool_epochs', type=int,
                           help='retraining epochs during trigger selection '
                           '(default: 600)')
        group.add_argument('--refool_lr', type=float,
                           help='retraining learning rate during trigger selection '
                           '(default: 1e-3)')
        group.add_argument('--pascal_root', help='path to Pascal VOC dataset '
                           '(default: "{data_dir}/image/pascal_voc")')
        return group

    def __init__(self, candidate_num: int = 200, select_iter: int = 16,
                 refool_epochs: int = 600, refool_lr: float = 1e-3,
                 pascal_root: str = None,
                 poison_percent: float = 0.4, **kwargs):
        super().__init__(poison_percent=poison_percent, **kwargs)
        self.param_list['reflection'] = ['candidate_num', 'select_iter', 'refool_epochs']
        self.candidate_num = candidate_num
        self.select_iter = select_iter
        self.refool_epochs = refool_epochs
        self.refool_lr = refool_lr

        if pascal_root is None:
            pascal_root = os.path.join(env['data_dir'], 'image', 'pascal_voc')
        self.pascal_root = pascal_root
        self.reflect_imgs = self.get_reflect_imgs()

        mark_shape = self.dataset.data_shape.copy()
        mark_shape[0] += 1
        self.mark.mark = torch.ones(mark_shape, device=self.mark.mark.device)
        self.mark.mark_height, self.mark.mark_width = self.dataset.data_shape[-2:]
        self.mark.mark_height_offset, self.mark.mark_width_offset = 0, 0
        self.mark.mark_random_init = False
        self.mark.mark_random_pos = False
        self.mark.mark_alpha = None     # TODO: any manual alpha setting?

        self.target_set = self.dataset.get_dataset('train', class_list=[self.target_class])
        self.poison_num = round(len(self.target_set) * self.poison_percent)
        self.poison_ratio = self.poison_ratio
        self.train_mode = 'dataset'

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if 'mark_alpha' not in kwargs.keys() and self.mark.mark_alpha is None:
            x_weight = x.mean()
            mark_weight = self.mark.mark.mean()
            alpha = x_weight / (x_weight + mark_weight)
            kwargs['mark_alpha'] = alpha
        return super().add_mark(x, **kwargs)

    def get_reflect_imgs(self) -> torch.Tensor:
        tar_path = os.path.join(self.pascal_root, 'reflect.tar')
        if not os.path.isfile(tar_path):
            self.gen_reflect_imgs(tar_path, self.pascal_root, num_attack=self.candidate_num)
        tf = tarfile.open(tar_path, mode='r')
        transform = transforms.Compose([
            transforms.Resize([self.dataset.data_shape[-2:]]),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        images = torch.stack([transform(Image.open(tf.extractfile(member), mode='r'))
                              for member in tf.getmembers()])
        tf.close()
        return images.to(device=env['device'])

    def attack(self, epochs: int, optimizer: torch.optim.Optimizer, **kwargs):
        model_dict = self.model.state_dict()
        W = torch.ones(len(self.reflect_imgs))
        refool_optimizer = torch.optim.SGD(optimizer.param_groups[0]['params'],
                                           lr=self.refool_lr, momentum=0.9,
                                           weight_decay=5e-4)
        logger = MetricLogger(meter_length=35)
        logger.create_meters(attack_succ_rate='{median:.3f} ({min:.3f}  {max:.3f})')
        for _ in logger.log_every(range(self.select_iter)):
            # prepare data
            idx = random.choices(range(len(W)), weights=W.tolist(), k=self.poison_num)
            clean_data = sample_batch(self.target_set, self.poison_num)
            poison_data = [self.add_mark(clean_data[i], mark=self.reflect_imgs[idx[i]]) for i in range(len(idx))]
            dataset = TensorListDataset(torch.stack(poison_data), [self.target_class] * len(poison_data))
            loader = self.dataset.get_dataloader(mode='train', dataset=dataset)
            # train
            self.model._train(self.refool_epochs, optimizer=refool_optimizer,
                              loader_train=loader, validate_interval=0, verbose=False)
            # test
            select_idx = list(set(idx))
            marks = self.reflect_imgs[select_idx]
            asr_result = self.get_asr_result(marks)
            # update W
            W[select_idx] = asr_result
            other_idx = list(set(range(len(W))) - set(idx))
            W[other_idx] = asr_result.median()

            logger.reset().update_list(attack_succ_rate=asr_result)
            self.model.load_state_dict(model_dict)
        self.mark.mark[:-1] = self.reflect_imgs[W.argmax().item()]
        super().attack(epochs=epochs, optimizer=optimizer, **kwargs)

    def get_asr_result(self, marks: torch.Tensor) -> torch.Tensor:
        asr_result_list = []
        for mark in marks:
            self.mark.mark[:-1] = mark
            _, target_acc = self.model._validate(get_data_fn=self.get_data, keep_org=False,
                                                 poison_label=True, verbose=False)
            # Original code considers an untargeted-like attack scenario.
            # _, org_acc = self.model._validate(get_data_fn=self.get_data, keep_org=False,
            #                                   poison_label=False, verbose=False)
            # target_acc = 100 - org_acc
            asr_result_list.append(target_acc)
        return torch.stack(asr_result_list)

    @classmethod
    def gen_reflect_imgs(cls, tar_path: str, pascal_root: str, num_attack: int = 160,
                         reflect_class: set[str] = {'cat'},
                         background_class: set[str] = {'person'}):
        print('get image paths')
        if not os.path.isdir(pascal_root):
            os.makedirs(pascal_root)
        datasets = [torchvision.datasets.VOCDetection(pascal_root, year=year, image_set=image_set,
                                                      download=True) for year, image_set in sets]
        background_paths = cls.get_img_paths(datasets, positive_class=background_class, negative_class=reflect_class)
        reflect_paths = cls.get_img_paths(datasets, positive_class=reflect_class, negative_class=background_class)
        print()
        print('background: ', len(background_paths))
        print('reflect: ', len(reflect_paths))
        print()
        print('load images')
        reflect_imgs = [read_tensor(fp) for fp in reflect_paths]

        print('writing tar file: ', tar_path)
        tf = tarfile.open(tar_path, mode='w')
        logger = MetricLogger(meter_length=35)
        logger.create_meters(reflect_num=f'[ {{count:3d}} / {num_attack:3d} ]',
                             succ_num='{count:3d}',
                             reflect_mean='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                             diff_mean='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                             blended_max='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                             ssim='{global_avg:.3f} ({min:.3f}  {max:.3f})')
        candidates: set[int] = set()
        for fp in logger.log_every(background_paths):
            background_img = read_tensor(fp)
            for i, reflect_img in enumerate(reflect_imgs):
                blended, background_layer, reflection_layer = cls.blend_images(
                    background_img, reflect_img, ghost_rate=0.39)
                reflect_mean: float = reflection_layer.mean().item()
                diff_mean: float = (blended - reflection_layer).mean().item()
                blended_max: float = blended.max().item()
                logger.update(reflect_mean=reflect_mean, diff_mean=diff_mean, blended_max=blended_max)
                if reflect_mean < 0.8 * diff_mean and blended_max > 0.1:
                    ssim: float = skimage.metrics.structural_similarity(
                        blended.numpy(), background_layer.numpy(), channel_axis=0)
                    logger.update(ssim=ssim)
                    if 0.7 < ssim < 0.85:
                        logger.update(succ_num=1)
                        if i not in candidates:
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
            if len(candidates) == num_attack:
                break
        tf.close()

    @staticmethod
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

    @staticmethod
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
