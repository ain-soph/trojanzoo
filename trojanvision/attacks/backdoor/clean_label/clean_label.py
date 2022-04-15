#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 20 --lr 0.01 --attack clean_label --tqdm
"""  # noqa: E501

from ...abstract import CleanLabelBackdoor

import torch.utils.data


class CleanLabel(CleanLabelBackdoor):
    def __init__(self, *args, train_mode: str = 'dataset', **kwargs):
        super().__init__(*args, train_mode=train_mode, **kwargs)
        self.poison_set = self.get_poison_dataset()

    def get_poison_dataset(self, poison_num: int = None,
                           seed: int = None
                           ) -> torch.utils.data.Dataset:
        return super().get_poison_dataset(poison_num, load_mark=False, seed=seed)
