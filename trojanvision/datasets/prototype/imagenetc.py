#!/usr/bin/env python3

from trojanvision.datasets.folder import ImageNet
from trojanvision.utils.datasets.prototype import ImageNetC as ImageNetCDataset


class ImageNetC(ImageNet):
    r"""ImageNet-C dataset.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * paper: `Benchmarking Neural Network Robustness to Common Corruptions and Perturbations`_
        * website: https://github.com/hendrycks/robustness/

    Attributes:
        name (str): ``'imagenetc'``
        num_classes (int): ``1000``
        data_shape (list[int]): ``[3, 224, 224]``
        norm_par (dict[str, list[float]]):
            ``{'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}``

    .. _Benchmarking Neural Network Robustness to Common Corruptions and Perturbations:
        https://arxiv.org/abs/1903.12261
    """

    name: str = 'imagenetc'
    num_classes: int = 1000
    data_shape = [3, 224, 224]

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        pass

    def _get_org_dataset(self, mode, **kwargs):
        if mode in ['train', 'valid']:
            return None
        return ImageNetCDataset(distortion_name=mode, **kwargs)
