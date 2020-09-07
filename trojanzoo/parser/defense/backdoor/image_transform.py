# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Image_Transform(Parser_Defense_Backdoor):
    r"""Image Transform Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'image_transform'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--transform_mode', dest='transform_mode', type=str,
                            help='Image Transform Mode, defaults to config[image_transform][transform_mode]="recompress".')
        parser.add_argument('--resize_ratio', dest='resize_ratio', type=float,
                            help='Image Resize Ratio for Recompress, defaults to config[image_transform][resize_ratio]=0.95.')
