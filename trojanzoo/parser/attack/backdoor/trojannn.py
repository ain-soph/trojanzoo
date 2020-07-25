# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_TrojanNN(Parser_BadNet):
    r"""Trojan Net Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'trojannn'``
    """
    attack = 'trojannn'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                            help='the chosen feature layer patched by trigger where rare neuron activation is maxmized, defaults to config[trojannn][preprocess_layer]=features')
        parser.add_argument('--threshold', dest='threshold', type=float,
                            help='Trojan Net Threshold, defaults to config[trojannn][threshold][dataset]=5')
        parser.add_argument('--target_value', dest='target_value', type=float,
                            help='Trojan Net Target_Value, defaults to config[trojannn][target_value][dataset]=10')
        parser.add_argument('--neuron_lr', dest='neuron_lr', type=float,
                            help='Trojan Net learning rate in neuron preprocessing, defaults to config[trojannn][target_value][dataset]=0.015')
        parser.add_argument('--neuron_epoch', dest='neuron_epoch', type=int,
                            help='Trojan Net epoch in neuron preprocessing, defaults to config[trojannn][neuron_epoch][dataset]=20')
        parser.add_argument('--neuron_num', dest='neuron_num', type=int,
                            help='Trojan Net neuron numbers in neuron preprocessing, defaults to config[trojannn][neuron_num][dataset]=2')
