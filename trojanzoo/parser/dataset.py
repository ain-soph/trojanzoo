# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.dataset import Dataset

from trojanzoo.config import Config
config = Config.config


class Parser_Dataset(Parser):
    """ Dataset Parser.

    :param name: ``'dataset'``.
    :type name: str
    """
    name = 'dataset'

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
    def get_module(cls, dataset: str = None, **kwargs) -> Dataset:
        """get dataset.

        :param dataset: dataset name, defaults to ``config['dataset']['default_dataset']``.
        :type dataset: str, optional
        :return: Dataset Instance
        :rtype: Dataset
        """
        if dataset is None:
            dataset: str = config['dataset']['default_dataset']
        result: Param = cls.combine_param(config=config['dataset'], dataset=dataset,
                                          filter_list=['default_dataset'], **kwargs)
        return super().get_module('dataset', dataset, **result)
