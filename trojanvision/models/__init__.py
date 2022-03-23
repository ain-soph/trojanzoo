#!/usr/bin/env python3

from .imagemodel import _ImageModel, ImageModel

from .nas import *
from .normal import *
from .others import *
from .torchvision import *

from . import nas, normal, others, torchvision

from trojanvision.configs import config
import trojanzoo.models

from trojanvision.datasets import ImageSet
from trojanzoo.configs import Config
import argparse

module_list = [nas, normal, others, torchvision]
__all__ = ['_ImageModel', 'ImageModel',
           'add_argument', 'create',
           'output_available_models']
class_dict: dict[str, type[ImageModel]] = {}
for module in module_list:
    __all__.extend(module.__all__)
    class_dict.update(module.class_dict)


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: str | ImageModel = None,
                 config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict):
    r"""
    | Add image model arguments to argument parser.
    | For specific arguments implementation, see :meth:`ImageModel.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        model_name (str): The model name.
        model (str | ImageModel): Model instance or model name
            (as the alias of `model_name`).
        config (Config): The default parameter config,
            which contains the default dataset and model name if not provided.
        class_dict (dict[str, type[ImageModel]]):
            Map from model name to model class.
            Defaults to ``trojanvision.models.class_dict``.

    See Also:
        :func:`trojanzoo.models.add_argument()`
    """
    return trojanzoo.models.add_argument(parser=parser, model_name=model_name, model=model,
                                         config=config, class_dict=class_dict)


def create(model_name: str = None, model: str | ImageModel = None,
           dataset_name: str = None, dataset: str | ImageSet = None,
           config: Config = config, class_dict: dict[str, type[ImageModel]] = class_dict,
           **kwargs) -> ImageModel:
    r"""
    | Create a model instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{model_dir}/{dataset.data_type}/{dataset.name}'``.
    | For model implementation, see :class:`ImageModel`.

    Args:
        model_name (str): The model name.
        model (str | ImageModel): The model instance or model name
            (as the alias of `model_name`).
        dataset_name (str): The dataset name.
        dataset (str | trojanvision.datasets.ImageSet):
            Dataset instance or dataset name
            (as the alias of `dataset_name`).
        config (Config): The default parameter config.
        class_dict (dict[str, type[ImageModel]]):
            Map from model name to model class.
            Defaults to ``trojanvision.models.class_dict``.
        **kwargs: Keyword arguments
            passed to model init method.

    Returns:
        ImageModel: The image model instance.

    See Also:
        :func:`trojanzoo.models.create()`
    """
    return trojanzoo.models.create(model_name=model_name, model=model,
                                   dataset_name=dataset_name, dataset=dataset,
                                   config=config, class_dict=class_dict, **kwargs)


def output_available_models(class_dict: dict[str, type[ImageModel]] = class_dict, indent: int = 0) -> None:
    r"""Output all available model names.

    Args:
        class_dict (dict[str, type[ImageModel]]): Map from model name to model class.
            Defaults to ``trojanvision.models.class_dict``.
        indent (int): The space indent for the entire string.
            Defaults to ``0``.

    See Also:
        :func:`trojanzoo.models.output_available_models()`
    """
    return trojanzoo.models.output_available_models(class_dict=class_dict, indent=indent)


def get_available_models(class_dict: dict[str, type[ImageModel]] = class_dict) -> dict[str, list[str]]:
    return trojanzoo.models.get_available_models(class_dict=class_dict)
