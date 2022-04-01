#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack badnet
"""  # noqa: E501

from ...abstract import BackdoorAttack


class BadNet(BackdoorAttack):
    r"""BadNet proposed by Tianyu Gu from New York University in 2017.

    It inherits :class:`trojanvision.attacks.BackdoorAttack` and actually equivalent to it.

    See Also:
        * paper: `BadNets\: Identifying Vulnerabilities in the Machine Learning Model Supply Chain`_

    .. _BadNets\: Identifying Vulnerabilities in the Machine Learning Model Supply Chain:
        https://arxiv.org/abs/1708.06733
    """
    name: str = 'badnet'
