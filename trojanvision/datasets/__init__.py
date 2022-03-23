#!/usr/bin/env python3

from .imageset import ImageSet
from .imagefolder import ImageFolder

from .folder import *
from .normal import *

from . import folder, normal

from trojanvision.configs import config
import trojanzoo.datasets

import argparse
from trojanzoo.configs import Config

module_list = [folder, normal]
__all__ = ['ImageSet', 'ImageFolder', 'class_dict', 'add_argument', 'create']
class_dict: dict[str, type[ImageSet]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)


def add_argument(parser: argparse.ArgumentParser,
                 dataset_name: str = None, dataset: str | ImageSet = None,
                 config: Config = config, class_dict: dict[str, type[ImageSet]] = class_dict
                 ) -> argparse._ArgumentGroup:
    r"""
    | Add image dataset arguments to argument parser.
    | For specific arguments implementation, see :meth:`ImageSet.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        dataset_name (str): The dataset name.
        dataset (str | Dataset): Dataset instance or dataset name
            (as the alias of `dataset_name`).
        config (Config): The default parameter config,
            which contains the default dataset name if not provided.
        class_dict (dict[str, type[Dataset]]):
            Map from dataset name to dataset class.
            Defaults to ``trojanvision.datasets.class_dict``.

    See Also:
        :func:`trojanzoo.datasets.add_argument()`
    """
    return trojanzoo.datasets.add_argument(parser=parser, dataset_name=dataset_name, dataset=dataset,
                                           config=config, class_dict=class_dict)


def create(dataset_name: str = None, dataset: str = None,
           config: Config = config, class_dict: dict[str, type[ImageSet]] = class_dict,
           **kwargs) -> ImageSet:
    r"""
    | Create a image dataset instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{data_dir}/{data_type}/{name}'``.
    | For dataset implementation, see :class:`ImageSet`.

    Args:
        dataset_name (str): The dataset name.
        dataset (str): The alias of `dataset_name`.
        config (Config): The default parameter config.
        class_dict (dict[str, type[ImageSet]]):
            Map from dataset name to dataset class.
            Defaults to ``trojanvision.datasets.class_dict``.
        **kwargs: Keyword arguments
            passed to dataset init method.

    Returns:
        ImageSet: Image dataset instance.

    See Also:
        :func:`trojanzoo.datasets.create()`
    """
    return trojanzoo.datasets.create(dataset_name=dataset_name, dataset=dataset,
                                     config=config, class_dict=class_dict, **kwargs)
