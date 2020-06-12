# -*- coding: utf-8 -*-

from ..parser import Parser
from trojanzoo.dataset import Dataset
from trojanzoo.attack import Attack

from trojanzoo.utils import Config
config = Config.config


class Parser_Attack(Parser):
    r"""Generic Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): The specific attack name (lower-case).
    """
    name = 'attack'
    attack = None

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--iteration', dest='iteration', type=int,
                            help='attack iterations, defaults to config[attack][iteration][dataset].')
        parser.add_argument('--early_stop', dest='early_stop', action='store_true',
                            help='early stop when reach stop_confidence.')
        parser.add_argument('--stop_confidence', dest='stop_confidence', type=float,
                            help='stop confidence for early stop, defaults to config[attack][stop_confidence][dataset].')
        parser.add_argument('--output', dest='output', type=int,
                            help='output level, defaults to config[attack][output][dataset].')

    @classmethod
    def get_module(cls, attack: str = None, dataset: Dataset = None, **kwargs) -> Attack:
        # type: (str, Dataset, dict) -> Attack  # noqa
        """get attack. specific attack config overrides general attack config.

        Args:
            attack (str): attack name
            dataset (Dataset):

        Returns:
            attack instance (:class:`Attack`).
        """
        if attack is None:
            attack = cls.attack
        result: Param = cls.combine_param(config=config['attack'],
                                          dataset=dataset)
        specific: Param = cls.combine_param(config=config[attack],
                                            dataset=dataset, **kwargs)
        result.update(specific)
        return super().get_module('attack', attack, **result)
