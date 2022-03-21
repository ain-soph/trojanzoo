#!/usr/bin/env python3

from ...abstract import InputFiltering
from trojanzoo.environ import env
from trojanzoo.utils.data import TensorListDataset
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.metric import mask_jaccard

import torch
import numpy as np
from sklearn.cluster import KMeans


class Neo(InputFiltering):
    r"""

    https://github.com/sakshiudeshi/Neo
    Note:
        Neo assumes the defender has the knowledge of the trigger size.
    """
    name: str = 'neo'

    def __init__(self, threshold_t: float = 80.0, k_means_num: int = 3, sample_num: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.param_list['neo'] = ['threshold_t', 'k_means_num', 'sample_num']
        self.threshold_t = threshold_t
        self.k_means_num = k_means_num
        self.sample_num = sample_num

        self.mark_size: list[int] = [self.attack.mark.mark_height, self.attack.mark.mark_width]
        self.select_num = self.mark_size[0] * self.mark_size[1]

    @torch.no_grad()
    def get_pred_labels(self) -> torch.Tensor:
        r"""Get predicted labels for test inputs.

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(2 * defense_input_num)``.
        """
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
            poison_input = self.attack.add_mark(_input)
            clean_list.append(self.get_pred_label(_input[0], logger=logger))
            poison_list.append(self.get_pred_label(poison_input[0], logger=logger))
        return torch.as_tensor(clean_list + poison_list, dtype=torch.bool)

    def get_pred_label(self, img: torch.Tensor, logger: MetricLogger = None) -> bool:
        # get dominant color
        dom_c = self.get_dominant_color(img).unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1)

        # generate random numbers
        height, width = img.shape[-2:]
        pos_height = torch.randint(
            low=0, high=height - self.mark_size[0], size=[self.sample_num, 1])
        pos_width = torch.randint(
            low=0, high=width - self.mark_size[1], size=[self.sample_num, 1])
        pos_list = torch.stack([pos_height, pos_width], dim=1)    # (sample_num, 2)
        # block potential triggers on _input
        block_input = img.repeat(self.sample_num, 1, 1, 1)  # (sample_num, C, H, W)
        for i in range(self.sample_num):
            x = pos_list[i][0]
            y = pos_list[i][1]
            block_input[i, :, x:x + self.mark_size[0], y:y + self.mark_size[1]] = dom_c
        # get potential triggers
        org_class = self.model.get_class(img.unsqueeze(0)).item()   # (1)
        block_class = self.model.get_class(block_input).cpu()   # (sample_num)

        # confirm triggers
        pos_pairs = pos_list[block_class != org_class]   # (*, 2)
        result = False
        for pos in pos_pairs:
            self.attack.mark.mark_height_offset = pos[0]
            self.attack.mark.mark_width_offset = pos[1]
            self.attack.mark.mark.fill_(1.0)
            self.attack.mark.mark[:-1] = img[..., pos[0]:pos[0] + self.mark_size[0],
                                             pos[1]:pos[1] + self.mark_size[1]]
            cls_diff = self.get_cls_diff()
            if cls_diff > self.threshold_t:
                result = True
                jaccard_idx = mask_jaccard(self.attack.mark.get_mask(),
                                           self.real_mask,
                                           select_num=self.select_num)
                logger.update(cls_diff=cls_diff, jaccard_idx=jaccard_idx)
        return result

    def get_dominant_color(self, img: torch.Tensor) -> torch.Tensor:
        r"""Get dominant color for one image tensor
        using :class:`sklearn.cluster.KMeans`.

        Args:
            img (torch.Tensor): Image tensor with shape ``(C, H, W)``.

        Returns:
            torch.Tensor: Dominant color tensor with shape ``(C)``.
        """
        img_np: np.ndarray = img.flatten(1).transpose(0, 1).cpu().numpy()    # (*, C)
        kmeans_result = KMeans(n_clusters=self.k_means_num).fit(img_np)
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
            diff.update(result.float().mean().mul(100).item(), len(_input))
        return diff.global_avg
