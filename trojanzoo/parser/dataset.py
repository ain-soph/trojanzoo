# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.dataset import Dataset

from trojanzoo.utils import Config
config = Config.config


class Parser_Dataset(Parser):
    r"""Dataset Parser.

    Attributes:
        name (str): ``'dataset'``
    """

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                            help='dataset name (lowercase).')
        parser.add_argument('--batch_size', dest='batch_size', type=int,
                            help='batch size (negative number means batch_size for each gpu).')
        parser.add_argument('--num_workers', dest='num_workers', type=int,
                            help='num_workers passed to torch.utils.data.DataLoader for training set, defaults to 4. (0 for validation set)')
        parser.add_argument('--download', dest='download', action='store_true',
                            help='download dataset if not exist by calling dataset.initialize()')

    @classmethod
    def get_module(cls, dataset=None, **kwargs):
        # type: (str, dict) -> Dataset  # noqa
        r"""get dataset.

        Args:
            dataset (str): dataset name, default: ``config['dataset']['default_dataset']``.
        Returns:
            :class:`Dataset`
        """
        if dataset is None:
            dataset: str = config['dataset']['default_dataset']
        result = cls.combine_param(config=config['dataset'], dataset=dataset,
                                   filter_list=['default_dataset'], **kwargs)
        return super().get_module('dataset', dataset, **result)
