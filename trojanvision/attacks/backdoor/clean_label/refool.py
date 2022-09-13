#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 20 --lr 0.01 --attack refool --tqdm --efficient
"""  # noqa: E501

from ...abstract import CleanLabelBackdoor
from trojanvision.environ import env
from trojanzoo.utils.data import TensorListDataset, sample_batch
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.output import output_iter

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

import math
import random
import skimage.metrics
import PIL.Image as Image

import copy
import io
import os
import tarfile
from xml.etree.ElementTree import parse as ET_parse

import argparse
import torch.utils.data

sets: list[tuple[str, str]] = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
norm = torch.distributions.normal.Normal(loc=0.0, scale=1.0)    # TODO: avoid construction when unused


def read_tensor(fp: str) -> torch.Tensor:
    tensor = F.convert_image_dtype(F.pil_to_tensor(Image.open(fp)))
    return tensor.unsqueeze(0) if tensor.dim() == 2 else tensor


class Refool(CleanLabelBackdoor):
    r"""Reflection Backdoor Attack (Refool) proposed by Yunfei Liu
    from Beihang University in ECCV 2020.

    It inherits :class:`trojanvision.attacks.CleanLabelBackdoor`.

    Note:
        * Trigger size must be the same as image size.
        * Currently, :attr:`mark_alpha` is forced to be ``-1.0``,
          which means to use mean of image and mark to blend them.
          It should be possible to set a manual :attr:`mark_alpha` instead.

    The attack has 3 procedures:

        * **Generate** :attr:`candidate_num` **reflect images from another public dataset (e.g., Pascal VOC) as trigger candidates.**

          - Select a :attr:`reflect class` (e.g., ``'cat'``)
            and a :attr:`background class` (e.g., ``'person'``)
          - Find all images of those 2 classes that
            don't have the object of the other class in them.
          - For image pairs from 2 classes, process and blend them using ``'ghost effect'``
            or ``'focal blur'``.
          - Calculate difference between blended image and reflect image.
          - Calculate structure similarity (SSIM) between blended image and background image
            by calling :any:`skimage.metrics.structural_similarity`.
          - If the difference is relatively large enough, blended image is not very dark
            and SSIM is around ``(0.7, 0.85)``, current reflect image is added to candidates.
        * **Rank candidate triggers by conducting tentative attack with multiple triggers injected together.**

          - (Initialize, not repeated) Assign all candidate triggers with same sampling weights.
          - Sample certain amount (e.g., ``40%`` in original code) of clean data from training set in target class.
          - Randomly attach a candidate trigger on each clean input according to their sampling weights.
          - Use the infected data as poison dataset to retrain a pretrained model
            with :attr:`refool_epochs` and :attr:`refool_lr`.
          - Evaluate attack succ rate of each used trigger as their new sampling weights.
          - Set sampling weights of all unused triggers to the median of used ones.
          - Reset the model as pretrained state.
          - Repeat the ranking process for :attr:`rank_iter` times.
        * **Use the trigger with largest sampling weight for final attack**
          (with ``'dataset'`` train_mode).

    See Also:
        * paper: `Reflection Backdoor\: A Natural Backdoor Attack on Deep Neural Networks`_
        * code: https://github.com/DreamtaleCore/Refool

    Note:
        There are **differences** between our implementation and original codes.
        I've consulted first author to clarify that current implementation of TrojanZoo should work.

        * | Author's code allows repeat during generating candidate reflect images.
          | Our code has **NO** repeat.
        * | Author's code generates ``160`` (actually usually not reaching this number)
            candidate reflect images but requires ``200`` during attack, which causes more repeat.
          | Our code generate :attr:`candidate_num` (``100`` as default) unique candidates.
        * | Author's code uses a very large :attr:`refool_epochs` (``600``),
            which causes too much clean accuracy drop and is very slow.
          | Our code uses ``5`` as default.
        * | Author's code uses a very large :attr:`refool_sample_percent` (``0.4``),
            which causes too much clean accuracy drop.
          | Our code uses ``0.1`` as default.
        * | There should be a pretrained model that is reset at every ranking loop.
          | However, the paper and original code don't mention that.
          | The author tells me that they load pretrained model from ImageNet.
        * There is no attack code provided by original author after ranking candidate reflect images.

        There is also a **conflict** between codes and paper from original author.

        * | Paper claims to use top-:attr:`candidate_num` selection at every ranking loop in Algorithm 1.
          | Author's code uses random sampling according to ``W`` as sampling weights.
          | Our code follows **author's code**.

    Args:
        candidate_num (int): Number of candidate reflect images.
            Defaults to ``100``.
        rank_iter (int): Iteration to update sampling weights of candidate reflect images.
            Defaults to ``16``.
        refool_epochs (int): Retraining epochs during trigger ranking.
            Defaults to ``5``.
        refool_lr (float): Retraining learning rate during trigger ranking.
            Defaults to ``1e-3``.
        refool_sample_percent (float): Percentage of retraining samples
            by training set in target class during trigger ranking.
            Defaults to ``0.1``.
        voc_root (str): Path to Pascal VOC dataset.
            Defaults to ``'{data_dir}/image/voc'``.
        efficient (bool): Whether to only use a subset (20%) to evaluate ASR during trigger ranking.
            Defaults to ``False``.

    Attributes:
        reflect_imgs (torch.Tensor): Candidate reflect images with shape ``(candidate_num, C, H, W)``.
        train_mode (str): Training mode to inject backdoor. Forced to be 'dataset'.
            See detailed description in :class:`trojanvision.attacks.BadNet`.
        poison_set (torch.utils.data.Dataset): Poison dataset (no clean data).
            It is ``None`` at initialization because the best trigger keeps unknown.
        refool_sample_num (int): Number of retraining samples from training set
            in target class during trigger ranking.
            ``refool_sample_percent * len(target_set)``
        target_set (torch.utils.data.Dataset): Training set in target class.

    .. _Reflection Backdoor\: A Natural Backdoor Attack on Deep Neural Networks:
        https://arxiv.org/abs/2007.02343
    """  # noqa: E501
    name: str = 'refool'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--candidate_num', type=int,
                           help='number of candidate reflect images '
                           '(default: 100)')
        group.add_argument('--rank_iter', type=int,
                           help='iteration to update sampling weights of candidate reflect images '
                           '(default: 16)')
        group.add_argument('--refool_epochs', type=int,
                           help='retraining epochs during trigger ranking '
                           '(default: 5)')
        group.add_argument('--refool_lr', type=float,
                           help='retraining learning rate during trigger ranking '
                           '(default: 1e-3)')
        group.add_argument('--refool_sample_percent', type=int,
                           help='retraining samples by training set '
                           'in target class during trigger ranking '
                           '(default: 0.1)')
        group.add_argument('--voc_root', help='path to Pascal VOC dataset '
                           '(default: "{data_dir}/image/voc")')
        group.add_argument('--efficient', action='store_true',
                           help='whether to only use a subset (20%) '
                           'to evaluate ASR during trigger ranking')
        return group

    def __init__(self, candidate_num: int = 100, rank_iter: int = 16,
                 refool_epochs: int = 5, refool_lr: float = 1e-3,
                 refool_sample_percent: float = 0.1,
                 voc_root: str = None, efficient: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['refool'] = ['candidate_num', 'rank_iter', 'refool_epochs']
        self.candidate_num = candidate_num
        self.rank_iter = rank_iter
        self.refool_epochs = refool_epochs
        self.refool_lr = refool_lr
        self.refool_sample_percent = refool_sample_percent

        if voc_root is None:
            data_dir = os.path.dirname(os.path.dirname(self.dataset.folder_path))
            voc_root = os.path.join(data_dir, 'image', 'voc')
        self.voc_root = voc_root
        self.reflect_imgs = self._get_reflect_imgs()

        mark_shape = self.dataset.data_shape.copy()
        mark_shape[0] += 1
        self.mark.mark = torch.ones(mark_shape, device=self.mark.mark.device)
        self.mark.mark_height, self.mark.mark_width = self.dataset.data_shape[-2:]
        self.mark.mark_height_offset, self.mark.mark_width_offset = 0, 0
        self.mark.mark_random_init = False
        self.mark.mark_random_pos = False
        self.mark.mark_alpha = -1.0     # TODO: any manual alpha setting?

        self.refool_sample_num = int(self.refool_sample_percent * len(self.target_set))

        if efficient:
            valid_set = self.dataset.loader['valid'].dataset
            length = int(0.2 * len(valid_set))
            subset = torch.utils.data.Subset(valid_set, torch.randperm(len(valid_set))[:length])
            self.loader_valid = self.dataset.get_dataloader(mode='valid', dataset=subset)
        else:
            self.loader_valid = self.dataset.loader['valid']

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor by calling
        :meth:`trojanvision.attacks.BadNet.add_mark()`.

        If :attr:`mark_alpha` ``<0``, use mean of :attr:`x`
        and :attr:`self.mark.mark` as their weights.
        """
        mark_alpha = kwargs.get('mark_alpha', self.mark.mark_alpha)
        if mark_alpha < 0:
            x_weight: float = x.mean().item()
            mark_weight: float = self.mark.mark[:-1].mean().item()
            alpha = mark_weight / (x_weight + mark_weight)
            kwargs['mark_alpha'] = alpha
        return super().add_mark(x, **kwargs)

    def attack(self, epochs: int, optimizer: torch.optim.Optimizer, **kwargs):
        model_dict = copy.deepcopy(self.model.state_dict())
        W = torch.ones(len(self.reflect_imgs))
        refool_optimizer = torch.optim.SGD(optimizer.param_groups[0]['params'],
                                           lr=self.refool_lr, momentum=0.9,
                                           weight_decay=5e-4)
        # logger = MetricLogger(meter_length=35)
        # logger.create_meters(asr='{median:.3f} ({min:.3f}  {max:.3f})')
        # iterator = logger.log_every(range(self.rank_iter))
        for _iter in range(self.rank_iter):
            print('Select iteration: ', output_iter(_iter + 1, self.rank_iter))
            # prepare data
            idx = random.choices(range(len(W)), weights=W.tolist(), k=self.refool_sample_num)
            mark = torch.ones_like(self.mark.mark).expand(self.refool_sample_num, -1, -1, -1).clone()
            mark[:, :-1] = self.reflect_imgs[idx]
            clean_input, _ = sample_batch(self.target_set, self.refool_sample_num)
            trigger_input = self.add_mark(clean_input, mark=mark)
            dataset = TensorListDataset(trigger_input, [self.target_class] * len(trigger_input))
            loader = self.dataset.get_dataloader(mode='train', dataset=dataset)
            # train
            self.model._train(self.refool_epochs, optimizer=refool_optimizer,
                              loader_train=loader, validate_interval=0,
                              output_freq='epoch', indent=4)
            self.model._validate(indent=4)
            # test
            select_idx = list(set(idx))
            marks = self.reflect_imgs[select_idx]
            asr_result = self._get_asr_result(marks)
            # update W
            W[select_idx] = asr_result
            other_idx = list(set(range(len(W))) - set(idx))
            W[other_idx] = asr_result.median()

            # logger.reset().update_list(asr=asr_result)
            self.model.load_state_dict(model_dict)
        self.mark.mark[:-1] = self.reflect_imgs[W.argmax().item()]
        self.poison_set = self.get_poison_dataset(load_mark=False)
        return super().attack(epochs=epochs, optimizer=optimizer, **kwargs)

    def _get_asr_result(self, marks: torch.Tensor) -> torch.Tensor:
        r"""Get attack succ rate result for each mark in :attr:`marks`.

        Args:
            marks (torch.Tensor): Marks tensor with shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Attack succ rate tensor with shape ``(N)``.
        """
        asr_list = []
        logger = MetricLogger(meter_length=35, indent=4)
        logger.create_meters(asr='{median:.3f} ({min:.3f}  {max:.3f})')
        for mark in logger.log_every(marks, header='mark', tqdm_header='mark'):
            self.mark.mark[:-1] = mark
            asr, _ = self.model._validate(get_data_fn=self.get_data, keep_org=False,
                                          poison_label=True, verbose=False,
                                          loader=self.loader_valid)
            # Original code considers an untargeted-like attack scenario.
            # org_acc, _ = self.model._validate(get_data_fn=self.get_data, keep_org=False,
            #                                   poison_label=False, verbose=False)
            # asr = 100 - org_acc
            logger.update(asr=asr)
            asr_list.append(asr)
        return torch.tensor(asr_list)

    def _get_reflect_imgs(self, force_regen: bool = False) -> torch.Tensor:
        r"""Get reflect images with shape ``(candidate_num, C, H, W)``.

        Will generate tar file containing reflect images
        if it doesn't exist or ``force_regen == True``.

        Args:
            force_regen (bool): Whether to force regenerating tar file.
                Defaults to ``False``.

        Returns:
            torch.Tensor: Reflect images with shape ``(N, C, H, W)``.
        """
        tar_path = os.path.join(self.voc_root, 'reflect.tar')
        if force_regen or not os.path.isfile(tar_path):
            gen_reflect_imgs(tar_path, self.voc_root, num_attack=self.candidate_num)
        tf = tarfile.open(tar_path, mode='r')
        transform = transforms.Compose([
            transforms.Resize([self.dataset.data_shape[-2:]]),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        images = torch.stack([transform(Image.open(tf.extractfile(member), mode='r'))
                              for member in tf.getmembers()])
        if len(images) >= self.candidate_num:
            images = images[:self.candidate_num]
        elif not force_regen:
            return self._get_reflect_imgs(force_regen=True)
        else:
            raise RuntimeError('Can not generate enough images')
        tf.close()
        return images.to(device=env['device'])


def gen_reflect_imgs(tar_path: str, voc_root: str, num_attack: int = 160,
                     reflect_class: set[str] = {'cat'},
                     background_class: set[str] = {'person'}):
    r"""Generate a tar file containing reflect images.

    Args:
        tar_path (str): Tar file path to save.
        voc_root (str): VOC dataset root path.
        num_attack (int): Number of reflect images to generate.
        reflect_class (set[str]): Set of reflect classes.
        background_class (set[str]): Set of background classes.
    """
    print('get image paths')
    if not os.path.isdir(voc_root):
        os.makedirs(voc_root)
    datasets = [torchvision.datasets.VOCDetection(voc_root, year=year, image_set=image_set,
                                                  download=True) for year, image_set in sets]
    background_paths = _get_img_paths(datasets, positive_class=background_class, negative_class=reflect_class)
    reflect_paths = _get_img_paths(datasets, positive_class=reflect_class, negative_class=background_class)
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
                         reflect_mean='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         diff_mean='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         blended_max='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         ssim='{global_avg:.3f} ({min:.3f}  {max:.3f})')
    candidates: set[int] = set()
    for fp in logger.log_every(background_paths):
        background_img = read_tensor(fp)
        for i, reflect_img in enumerate(reflect_imgs):
            if i in candidates:
                continue
            blended, background_layer, reflection_layer = blend_images(
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
    else:
        raise RuntimeError('Can not generate enough images')
    tf.close()


def _get_img_paths(datasets: list[torchvision.datasets.VOCDetection],
                   positive_class: set[str], negative_class: set[str]
                   ) -> list[str]:
    r"""Get image paths that contain at least 1 object in :attr:`positive_class`
    and no object in :attr:`negative_class`.

    Args:
        datasets: (list[torchvision.datasets.VOCDetection]):
            list of different VOC datasets.
        positive_class (set[str]): Selected image should contain
            at least 1 object in :attr:`positive_class`.
        negative_class (set[str]): Selected image should **NOT** contain
            any object in :attr:`negative_class`.

    Returns:
        list[str]: List of selected image paths.
    """
    image_paths: list[str] = []
    for dataset in datasets:
        for index in range(len(dataset)):
            target = dataset.parse_voc_xml(ET_parse(dataset.annotations[index]).getroot())
            label_names: set[str] = {obj['name'] for obj in target['annotation']['object']}
            if len(positive_class & label_names) != 0 and len(negative_class & label_names) == 0:
                image_paths.append(dataset.images[index])
    return image_paths


def blend_images(background_img: torch.Tensor, reflect_img: torch.Tensor,
                 max_image_size: int = 560, ghost_rate: float = 0.49, alpha_bg: float = None,
                 offset: tuple[int, int] = (0, 0), sigma: float = None, ghost_alpha: float = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend background layer and reflection layer together (including blurred & ghosted reflection layer).

    :attr:`background_img` is resized using :attr:`max_image_size`
    and :attr:`reflect_img` is resized to the same shape.

    Note:
        This blend method is only used to generate reflect images.
        To add watermark on images, please call :meth:`add_mark()`.

    Args:
        background_img (torch.Tensor): Background image tensor with shape ``([N], C, H, W)``.
        reflect_img (torch.Tensor): Reflect image tensor with shape ``([N], C, H', W')``.
        max_image_size (int): Max image size (the longer edge of height/width).
            :attr:`background_img` will be resized while keeping aspect ratio.
            Defaults to ``560``.
        ghost_rate (float): Probability to generate the blended image with ghost effect.
            Defaults to ``0.49``.
        alpha_bg (float): Weight of background image during blending.
            Defaults to ``1 - random.uniform(0.05, 0.45)``。
        offset (tuple[int, int]): Offset of height and width used in ghost effect.
            Defaults to ``(random.randint(3, 8), random.randint(3, 8))``。
        sigma (float): Gaussian kernel standard deviation.
            Defaults to ``random.uniform(1, 5)``.
        ghost_alpha (float): Weight of the first ghost image used in ghost effect.
            Defaults to ``abs(round(random.random()) - random.uniform(0.15, 0.5))``.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):
            ``blended, background_layer, reflection_layer``
            with shape ``([N], C, H, W)``.
    """
    if alpha_bg is None:
        alpha_bg = 1. - random.uniform(0.05, 0.45)
    h, w = background_img.shape[-2:]
    aspect_ratio = w / h
    h, w = (max_image_size, int(round(max_image_size * aspect_ratio))) if h > w \
        else (int(round(max_image_size / aspect_ratio)), max_image_size)
    # Original code uses cv2 INTER_CUBIC, which is slightly different from BICUBIC
    background_img = F.resize(background_img, size=(h, w), interpolation=InterpolationMode.BICUBIC).clamp(0, 1)
    reflect_img = F.resize(reflect_img, size=(h, w), interpolation=InterpolationMode.BICUBIC).clamp(0, 1)
    background_img.pow_(2.2)
    reflect_img.pow_(2.2)

    background_mask = alpha_bg * background_img
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

        reflect_mask = (1 - alpha_bg) * reflect_ghost
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
        g_mask = gen_kernel(max_image_size, 3).repeat(3, 1, 1)  # TODO: try to avoid hard encode 3 as channel
        alpha_r = (1 - alpha_bg / 2) * g_mask[..., new_h: new_h + h, new_w: new_w + w]

        reflect_mask = alpha_r * reflect_blur
        reflection_layer = (min(1., 4 * (1 - alpha_bg)) * reflect_mask).pow(1 / 2.2)

    blended = (reflect_mask + background_mask).pow(1 / 2.2)
    background_layer = background_mask.pow(1 / 2.2)
    return blended, background_layer, reflection_layer
