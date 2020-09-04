# -*- coding: utf-8 -*-

from ..defense_backdoor import Parser_Defense_Backdoor


class Parser_Activation_Clustering(Parser_Defense_Backdoor):
    r"""Activation Clustering Parser

    Attributes:
        name (str): ``'defense'``
        defense (str): The specific defense name (lower-case).
    """
    name: str = 'defense'
    defense = 'activation_clustering'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--mix_image_num', dest='mix_image_num', type=int,
                            help='the number of sampled image')

        parser.add_argument('--clean_image_ratio', dest='clean_image_ratio', type=float,
                            help='the ratio of clean image')

        parser.add_argument('--retrain_epoch', dest='retrain_epoch', type=int,
                            help='the epoch of retraining the model')
        
        parser.add_argument('--nb_clusters', dest='nb_clusters', type=int,
                            help='')
        parser.add_argument('--clustering_method', dest='clustering_method', type=str,
                            help='the amount of clusters')
        parser.add_argument('--nb_dims', dest='nb_dims', type=int,help='the dimension set in the process of reduceing the dimensionality of data')

        parser.add_argument('--reduce_method', dest='reduce_method', type=str,
                            help=' the method for reduing the dimensionality of data')
        parser.add_argument('--cluster_analysis', dest='cluster_analysis', type=str,
                            help='the method chosen to analyze whether cluster is the poison cluster, including size, distance, relative-size, silhouette-scores')