# -*- coding: utf-8 -*-

from .parser import Parser
from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.utils.model import split_name

from trojanzoo.utils.config import Config
config = Config.config


class Parser_Model(Parser):
    r"""Model Parser

    Attributes:
        name (str): ``'model'``
    """
    name: str = 'model'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('-m', '--model', dest='model',
                            help='model name, defaults to config[model][default_model][dataset]')
        parser.add_argument('--layer', dest='layer', type=int,
                            help='layer (optional, maybe embedded in --model)')
        parser.add_argument('--suffix', dest='suffix',
                            help='model name suffix, e.g. _adv_train')
        parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                            help='load pretrained weights')
        parser.add_argument('--official', dest='official', action='store_true',
                            help='load official weights')
        parser.add_argument('--sgm', dest='sgm', action='store_true',
                            help='whether to use sgm gradient, defaults to False')
        parser.add_argument('--sgm_gamma', dest='sgm_gamma', type=float,
                            help='sgm gamma, defaults to config[model][sgm_gamma][dataset]=1.0')

    @classmethod
    def get_module(cls, model: str = None, layer: int = None, dataset: Dataset = None, **kwargs) -> Model:
        # type: (str, int, Dataset, dict) -> Model  # noqa
        r"""get model.

        Args:
            model (str): model name. Default: None.
            layer (int): layer (optional, maybe embedded in ``model``). Default: None.
            dataset (Dataset): dataset. Default: None.

        Returns:
            :class:`Model`
        """
        if model is None:
            dataset_name: str = 'default'
            if isinstance(dataset, Dataset):
                dataset_name = dataset.name
            elif isinstance(dataset, str):
                dataset_name = dataset
            model: str = config['model']['default_model'][dataset_name]
        model, layer = split_name(model, layer=layer)

        result: Param = cls.combine_param(config=config['model'], dataset=dataset,
                                          filter_list=['default_model'], layer=layer, **kwargs)
        return super().get_module('model', model, **result)
