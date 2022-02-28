#!/usr/bin/env python3

from ..abstract import BackdoorAttack


class BadNet(BackdoorAttack):
    r"""BadNet proposed by Tianyu Gu from New York University in 2017.
    It inherits :class:`trojanvision.attacks.BackdoorAttack` and actually equivalent to it.

    It attaches a provided watermark to some training images and inject them into training set with target label.
    After retraining, the model will classify images with watermark of certain/all classes into target class.

    See Also:
        * paper: `BadNets\: Identifying Vulnerabilities in the Machine Learning Model Supply Chain`_

    .. _BadNets\: Identifying Vulnerabilities in the Machine Learning Model Supply Chain:
        https://arxiv.org/abs/1708.06733
    """

    name: str = 'badnet'
