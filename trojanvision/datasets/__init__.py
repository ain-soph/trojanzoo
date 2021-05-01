#!/usr/bin/env python3

from .imageset import ImageSet
from .imagefolder import ImageFolder

from .folder import *
from .normal import *

from . import folder, normal

from trojanvision.configs import Config, config
import trojanzoo.datasets

import argparse
from typing import Union

module_list = [folder, normal]
__all__ = ['ImageSet', 'ImageFolder', 'class_dict', 'add_argument', 'create']
class_dict: dict[str, type[ImageSet]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)


def add_argument(parser: argparse.ArgumentParser, dataset_name: str = None, dataset: Union[str, ImageSet] = None,
                 config: Config = config, class_dict: dict[str, type[ImageSet]] = class_dict) -> argparse._ArgumentGroup:
    return trojanzoo.datasets.add_argument(parser=parser, dataset_name=dataset_name, dataset=dataset,
                                           config=config, class_dict=class_dict)


def create(dataset_name: str = None, dataset: str = None,
           config: Config = config, class_dict: dict[str, type[ImageSet]] = class_dict, **kwargs) -> ImageSet:
    return trojanzoo.datasets.create(dataset_name=dataset_name, dataset=dataset,
                                     config=config, class_dict=class_dict, **kwargs)
