# -*- coding: utf-8 -*-

from .. import Parser

from trojanzoo.dataset import Dataset

from trojanzoo.utils import Config
config = Config.config


class Parser_Defense(Parser):
    r"""Generic Defense Parser

    Attributes:
        name (str): ``'defense'``
    """
    name: str = 'defense'
    defense = None

    @classmethod
    def get_module(cls, defense: str = None, dataset: Dataset = None, **kwargs):
        # type: (str, Dataset, dict)  # noqa
        """get defense. specific defense config overrides general defense config.

        Args:
            defense (str): defense name
            dataset (Dataset):

        Returns:
            defense instance.
        """
        if defense is None:
            defense = cls.defense
        result: Param = cls.combine_param(config=config[defense],
                                          dataset=dataset, **kwargs)
        return super().get_module('defense', defense, **result)
