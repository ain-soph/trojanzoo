# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Spectral_Signature(Parser_Defense_Backdoor):
    r"""Spectral Signature Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'spectral_signature'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                            help='the chosen layer used to extract feature representation, defaults to config[Spectral_Signature][preprocess_layer]=feature')


        parser.add_argument('--poison_image_num', dest='poison_image_num', type=int,
                            help='the number of sampled poison image to train the model initially, faults to config[Spectral_Signature][poison_image_num]=50')

        parser.add_argument('--clean_image_num', dest='clean_image_num', type=int,
                            help='the number of sampled clean image to train the model initially, faults to config[Spectral_Signature][clean_image_num]=500')

        parser.add_argument('--epsilon', dest='epsilon', type=int,
                            help='the number of examples to remove from each class, faults to config[Spectral_Signature][epsilon]=5')
        parser.add_argument('--retrain_epoch', dest='retrain_epoch', type=int,
                            help='the epoch to retrain the model on clean image dataset, faults to config[Spectral_Signature][retrain_epoch]=5')