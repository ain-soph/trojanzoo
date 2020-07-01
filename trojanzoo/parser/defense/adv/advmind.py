# -*- coding: utf-8 -*-

from ..defense import Parser_Defense
from trojanzoo.dataset import Dataset
from trojanzoo.defense import AdvMind

from trojanzoo.utils import Config
config = Config.config


class Parser_AdvMind(Parser_Defense):
    r"""AdvMind Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name = 'defense'
    defense = 'advmind'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--output', dest='output', type=int,
                            help='output level, defaults to config[attack][output][dataset].')

        parser.add_argument('--alpha', dest='alpha', type=float,
                            help='PGD learning rate per step, defaults to config[pgd][alpha][dataset]=3.0/255')
        parser.add_argument('--epsilon', dest='epsilon', type=float,
                            help='Projection norm constraint, defaults to config[pgd][epsilon][dataset]=8.0/255')
        parser.add_argument('--iteration', dest='iteration', type=int,
                            help='Attack Iteration, defaults to config[pgd][iteration][dataset]=20')
        parser.add_argument('--target_idx', dest='target_idx', type=int,
                            help='Target label order in original classification, defaults to config[pgd][target_idx][dataset]=1 '
                                 + '(0 for untargeted attack, 1 for most possible class, -1 for most unpossible class)')

        parser.add_argument('--grad_method', dest='grad_method',
                            help='black box gradient estimation method, defaults to config[pgd][grad_method][dataset]=\'nes\'')
        parser.add_argument('--query_num', dest='query_num', type=int,
                            help='query numbers for black box gradient estimation, defaults to config[pgd][query_num][dataset]=100.')
        parser.add_argument('--sigma', dest='sigma', type=float,
                            help='gaussian sampling std for black box gradient estimation, defaults to config[pgd][sigma][dataset]=1e-3')

        parser.add_argument('--attack_adapt', dest='attack_adapt', action='store_true',
                            help='Adaptive attack to add fake queries.')
        parser.add_argument('--fake_percent', dest='fake_percent', type=float,
                            help='fake query percentage.')
        parser.add_argument('--dist', dest='dist', type=float,
                            help='fake query noise std.')
        parser.add_argument('--defend_adapt', dest='defend_adapt', action='store_true',
                            help='Robust location M-estimator.')
        parser.add_argument('--active', dest='active', action='store_true',
                            help='Proactive solicitation.')
        parser.add_argument('--active_percent', dest='active_percent', type=float,
                            help='Active gradient weight.')

    @classmethod
    def get_module(cls, defense: str = None, dataset: Dataset = None, **kwargs) -> AdvMind:
        # type: (str, Dataset, dict)  # noqa
        """load extra pgd parameters from config['pgd']."""
        if defense is None:
            defense = cls.defense
        result: Param = cls.combine_param(config=config['pgd'],
                                          dataset=dataset, **kwargs)
        specific: Param = cls.combine_param(config=config[defense],
                                            dataset=dataset, **kwargs)
        result.update(specific)
        return super().get_module('defense', defense, **result)
