# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.dataset import ImageSet
from trojanzoo.utils.mark import Watermark

from typing import List

from trojanzoo.utils.config import Config
config = Config.config


class Parser_Mark(Parser):
    r"""
    Watermark Parser to process watermark image.

    Resize Override Priority: height,width > height(width)_ratio > mark_ratio

    Attributes:
        name (str): ``'mark'``
    """
    name: str = 'mark'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--edge_color', dest='edge_color',
                            help='edge color in watermark image, defaults to \'auto\'.')
        parser.add_argument('--mark_path', dest='mark_path',
                            help='edge color in watermark image, defaults to trojanzoo/data/mark/apple_white.png.')
        parser.add_argument('--mark_alpha', dest='mark_alpha', type=float,
                            help='mark transparency, defaults to 0.0.')
        parser.add_argument('--height', dest='height', type=int,
                            help='mark height.')
        parser.add_argument('--width', dest='width', type=int,
                            help='mark width.')
        parser.add_argument('--height_ratio', dest='height_ratio', type=float,
                            help='mark height ratio.')
        parser.add_argument('--width_ratio', dest='width_ratio', type=float,
                            help='mark width ratio.')
        parser.add_argument('--mark_ratio', dest='mark_ratio', type=float,
                            help='mark ratio.')
        parser.add_argument('--height_offset', dest='height_offset', type=int,
                            help='height offset, defaults to 0')
        parser.add_argument('--width_offset', dest='width_offset', type=int,
                            help='width offset, defaults to 0')
        parser.add_argument('--random_pos', dest='random_pos', action='store_true',
                            help='Random offset Location for add_mark.')
        parser.add_argument('--random_init', dest='random_init', action='store_true',
                            help='random values for mark pixel.')
        parser.add_argument('--mark_distributed', dest='mark_distributed', action='store_true',
                            help='Distributed Mark.')

    @classmethod
    def get_module(cls, data_shape: List[int] = None, dataset: ImageSet = None, **kwargs) -> Watermark:
        # type: (List[int], ImageSet, dict) -> Watermark  # noqa
        """get watermark.

        Args:
            data_shape (List[int]): ``[C, H, W]``. Default: None.
            dataset (ImageSet): dataset. Default: None.

        Returns:
            :class:`Watermark`
        """
        if data_shape is None:
            assert isinstance(dataset, ImageSet)
            data_shape: list = [dataset.n_channel]
            data_shape.extend(dataset.n_dim)

        result: Param = cls.combine_param(config=config['mark'], dataset=dataset,
                                          data_shape=data_shape, **kwargs)

        return Watermark(**result)
