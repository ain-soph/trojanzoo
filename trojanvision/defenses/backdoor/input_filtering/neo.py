#!/usr/bin/env python3

from ...abstract import InputFiltering
from trojanzoo.environ import env
from trojanzoo.utils.data import TensorListDataset
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.metric import mask_jaccard

import torch
import numpy as np
from sklearn.cluster import KMeans

import argparse


class Neo(InputFiltering):
    r"""Neo proposed by Sakshi Udeshi
    from Singapore University of Technology and Design
    in 2019.

    It is a input filtering backdoor defense
    that inherits :class:`trojanvision.defenses.InputFiltering`.

    The defense procedure is:

    - For a test input, Neo generates its different variants
      with a random region masked by the input's dominant color
      using :any:`sklearn.cluster.KMeans`.
    - For each variant, if its classification is different,
      check if the pixels from masked region is a trigger
      by evaluating its ASR.
    - If ASR of any variant exceeds the :attr:`neo_asr_threshold`,
      the test input is regarded as poisoned.

    See Also:
        * paper: `Model Agnostic Defence against Backdoor Attacks in Machine Learning`_
        * code: https://github.com/sakshiudeshi/Neo

    Note:
        Neo assumes the defender has the knowledge of the trigger size.

    Args:
        neo_asr_threshold (float): ASR threshold.
            Defaults to ``0.8``.
        neo_kmeans_num (int): Number of KMean clusters.
            Defaults to ``3``.
        neo_sample_num (int): Number of sampled masked regions.
            Defaults to ``100``.

    Attributes:
        mark_size(tuple[int, int]): Watermark size ``(h, w)`` of ``self.attack.mark``.

    .. _Model Agnostic Defence against Backdoor Attacks in Machine Learning:
        https://arxiv.org/abs/1908.02203
    """
    name: str = 'neo'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--neo_asr_threshold', type=float,
                           help='ASR threshold for neo defense '
                           '(default: 0.8)')
        group.add_argument('--neo_kmeans_num', type=int,
                           help='number of k-mean clusters for neo defense '
                           '(default: 3)')
        group.add_argument('--neo_sample_num', type=int,
                           help='number of sampled masked regions '
                           '(default: 100)')
        return group

    def __init__(self, neo_asr_threshold: float = 0.8, neo_kmeans_num: int = 3,
                 neo_sample_num: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.param_list['neo'] = ['neo_asr_threshold', 'neo_kmeans_num', 'neo_sample_num']
        self.neo_asr_threshold = neo_asr_threshold
        self.neo_kmeans_num = neo_kmeans_num
        self.neo_sample_num = neo_sample_num

        self.mark_size: tuple[int, int] = [self.attack.mark.mark_height, self.attack.mark.mark_width]
        self.select_num = self.mark_size[0] * self.mark_size[1]

    @torch.no_grad()
    def get_pred_labels(self) -> torch.Tensor:
        logger = MetricLogger(meter_length=40)
        str_format = '{global_avg:5.3f} ({min:5.3f}, {max:5.3f})'
        logger.create_meters(cls_diff=str_format, jaccard_idx=str_format)
        test_set = TensorListDataset(self.test_input, self.test_label)
        test_loader = self.dataset.get_dataloader(mode='valid', dataset=test_set, batch_size=1)
        clean_list = []
        poison_list = []
        for data in logger.log_every(test_loader):
            _input: torch.Tensor = data[0]
            _input = _input.to(env['device'], non_blocking=True)
            trigger_input = self.attack.add_mark(_input)
            clean_list.append(self.get_pred_label(_input[0], logger=logger))
            poison_list.append(self.get_pred_label(trigger_input[0], logger=logger))
        return torch.as_tensor(clean_list + poison_list, dtype=torch.bool)

    def get_pred_label(self, img: torch.Tensor, logger: MetricLogger = None) -> bool:
        r"""Get the prediction label of one certain image (poisoned or not).

        Args:
            img (torch.Tensor): Image tensor (on GPU) with shape ``(C, H, W)``.
            logger (trojanzoo.utils.logger.MetricLogger):
                output logger.
                Defaults to ``None``.

        Returns:
            bool: Whether the image tensor :attr:`img` is poisoned.
        """
        # get dominant color
        dom_c = self.get_dominant_color(img).unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1)

        # generate random numbers
        height, width = img.shape[-2:]
        pos_height = torch.randint(
            low=0, high=height - self.mark_size[0], size=[self.neo_sample_num, 1])
        pos_width = torch.randint(
            low=0, high=width - self.mark_size[1], size=[self.neo_sample_num, 1])
        pos_list = torch.stack([pos_height, pos_width], dim=1)    # (neo_sample_num, 2)
        # block potential triggers on _input
        block_input = img.repeat(self.neo_sample_num, 1, 1, 1)  # (neo_sample_num, C, H, W)
        for i in range(self.neo_sample_num):
            x = pos_list[i][0]
            y = pos_list[i][1]
            block_input[i, :, x:x + self.mark_size[0], y:y + self.mark_size[1]] = dom_c
        # get potential triggers
        org_class = self.model.get_class(img.unsqueeze(0)).item()   # (1)
        block_class = self.model.get_class(block_input).cpu()   # (neo_sample_num)

        # confirm triggers
        pos_pairs = pos_list[block_class != org_class]   # (*, 2)
        for pos in pos_pairs:
            self.attack.mark.mark_height_offset = pos[0]
            self.attack.mark.mark_width_offset = pos[1]
            self.attack.mark.mark.fill_(1.0)
            self.attack.mark.mark[:-1] = img[..., pos[0]:pos[0] + self.mark_size[0],
                                             pos[1]:pos[1] + self.mark_size[1]]
            cls_diff = self.get_cls_diff()
            if cls_diff > self.neo_asr_threshold:
                jaccard_idx = mask_jaccard(self.attack.mark.get_mask(),
                                           self.real_mask,
                                           select_num=self.select_num)
                logger.update(cls_diff=cls_diff, jaccard_idx=jaccard_idx)
                return True
        return False

    def get_dominant_color(self, img: torch.Tensor) -> torch.Tensor:
        r"""Get dominant color for one image tensor
        using :class:`sklearn.cluster.KMeans`.

        Args:
            img (torch.Tensor): Image tensor with shape ``(C, H, W)``.

        Returns:
            torch.Tensor: Dominant color tensor with shape ``(C)``.
        """
        img_np: np.ndarray = img.flatten(1).transpose(0, 1).cpu().numpy()    # (*, C)
        kmeans_result = KMeans(n_clusters=self.neo_kmeans_num).fit(img_np)
        unique, counts = np.unique(kmeans_result.labels_, return_counts=True)
        center = kmeans_result.cluster_centers_[unique[np.argmax(counts)]]
        return torch.as_tensor(center)

    def get_cls_diff(self):
        r"""Get classification difference between
        original inputs and trigger inputs.

        Returns:
            float: Classification difference percentage.
        """
        diff = SmoothedValue()
        for data in self.dataset.loader['valid']:
            _input, _ = self.model.get_data(data)
            _class = self.model.get_class(_input)
            trigger_input = self.attack.add_mark(_input)
            trigger_class = self.model.get_class(trigger_input)
            result = _class.not_equal(trigger_class)
            diff.update(result.float().mean().item(), len(_input))
        return diff.global_avg
