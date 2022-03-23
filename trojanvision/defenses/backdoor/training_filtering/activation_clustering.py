#!/usr/bin/env python3

from ...abstract import TrainingFiltering
from trojanvision.environ import env

import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data


class ActivationClustering(TrainingFiltering):
    r"""Activation Clustering proposed by Bryant Chen
    from IBM Research in SafeAI@AAAI 2019.

    It is a training filtering backdoor defense
    that inherits :class:`trojanvision.defenses.TrainingFiltering`.

    Activation Clustering assumes in the target class,
    poisoned samples compose a separate cluster
    which is small or far from its own class center.

    The defense procedure is:

    * Get feature maps for samples
    * For samples from each class

        * Get dim-reduced feature maps for samples using
          :any:`sklearn.decomposition.FastICA` or
          :any:`sklearn.decomposition.PCA`.
        * Conduct clustering w.r.t. dim-reduced feature maps and get cluster classes for samples.
        * Detect poisoned cluster classes. All samples in that cluster are poisoned.
          Poisoned samples compose a small separate class.

    There are 4 different methods to detect poisoned cluster classes:

    * ``'size'``: The smallest cluster class.
    * ``'relative size'``: The small cluster classes whose proportion is smaller than :attr:`size_threshold`.
    * ``'silhouette_score'``: only detect poison clusters using ``'relative_size'``
      when clustering fits data well.
    * ``'distance'``: Poison clusters are far from their own class center,

    See Also:
        * Paper: `Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering`_
        * Other implementation: `IBM adversarial robustness toolbox (ART)`_ [`source code`_]

    Args:
        nb_clusters (int): Number of clusters. Defaults to ``2``.
        nb_dims (int): The reduced dimension of feature maps. Defaults to ``10``.
        reduce_method (str): The method to reduce dimension of feature maps. Defaults to ``'FastICA'``.
        cluster_analysis (str): The method chosen to detect poisoned cluster classes.
            Choose from ``['size', 'relative_size', 'distance', 'silhouette_score']``
            Defaults to ``'silhouette_score'``.

    Note:
        Clustering method is :any:`sklearn.cluster.KMeans`
        if ``self.defense_input_num=None`` (full training set)
        else :any:`sklearn.cluster.MiniBatchKMeans`

    .. _Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering:
        https://arxiv.org/abs/1811.03728
    .. _IBM adversarial robustness toolbox (ART):
        https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/detector_poisoning.html#art.defences.detector.poison.ActivationDefence
    .. _source code:
        https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py
    """  # noqa: E501

    name: str = 'activation_clustering'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--nb_clusters', type=int,
                           help='number of clusters (default: 2)')
        group.add_argument('--nb_dims', type=int,
                           help='the reduced dimension of feature maps (default: 10)')
        group.add_argument('--reduce_method',
                           help='the method to reduce dimension of feature maps '
                           '(default: "FastICA")')
        group.add_argument('--cluster_analysis', choices=['size', 'relative_size', 'distance', 'silhouette_score'],
                           help='the method chosen to detect poisoned cluster classes '
                           '(default: "silhouette_score")')
        return group

    def __init__(self, nb_clusters: int = 2, nb_dims: int = 10,
                 reduce_method: str = 'FastICA',
                 cluster_analysis: str = 'silhouette_score',
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['activation_clustering'] = ['nb_clusters', 'nb_dims', 'reduce_method', 'cluster_analysis']

        self.nb_clusters = nb_clusters
        self.nb_dims = nb_dims
        self.reduce_method = reduce_method
        self.cluster_analysis = cluster_analysis

        match self.reduce_method:
            case 'FastICA':
                self.projector = FastICA(n_components=self.nb_dims)
            case 'PCA':
                self.projector = PCA(n_components=self.nb_dims)
            case _:
                raise ValueError(self.reduce_method + ' dimensionality reduction method not supported.')
        clusterer_class = MiniBatchKMeans if self.defense_input_num else KMeans
        self.clusterer = clusterer_class(n_clusters=self.nb_clusters)

    def get_pred_labels(self) -> torch.Tensor:
        all_fm = []
        all_pred_label = []
        mix_dataset = torch.utils.data.ConcatDataset([self.clean_set, self.poison_set])
        loader = self.dataset.get_dataloader('train', dataset=mix_dataset)
        if env['tqdm']:
            loader = tqdm(loader, leave=False)
        for data in loader:
            _input, _label = self.model.get_data(data)
            fm = self.model._model.get_final_fm(_input)
            pred_label = self.model.get_class(_input)
            all_fm.append(fm.detach().cpu())
            all_pred_label.append(pred_label.detach().cpu())
        all_fm = torch.cat(all_fm)
        all_pred_label = torch.cat(all_pred_label)

        result = torch.zeros_like(all_pred_label, dtype=torch.bool)
        analyze_func: Callable[..., list[int]] = getattr(self, f'analyze_by_{self.cluster_analysis}')

        idx_list: list[torch.Tensor] = []
        reduced_fm_centers_list: list[torch.Tensor] = []
        kwargs_list: list[dict[str, torch.Tensor]] = []
        for _class in range(self.dataset.num_classes):
            idx = all_pred_label == _class
            fm = all_fm[idx]
            reduced_fm = torch.as_tensor(self.projector.fit_transform(fm.numpy()))
            cluster_class = torch.as_tensor(self.clusterer.fit_predict(reduced_fm))
            kwargs_list.append(dict(cluster_class=cluster_class, reduced_fm=reduced_fm))
            idx_list.append(idx)
            if self.cluster_analysis == 'distance':
                reduced_fm_centers_list.append(reduced_fm.median(dim=0))
        if self.cluster_analysis == 'distance':
            reduced_fm_centers = torch.stack(reduced_fm_centers_list)

        for _class in range(self.dataset.num_classes):
            kwargs = kwargs_list[_class]
            idx = torch.arange(len(all_pred_label))[idx_list[_class]]
            if self.cluster_analysis == 'distance':
                kwargs['reduced_fm_centers'] = reduced_fm_centers
            poison_cluster_classes = analyze_func(_class=_class, **kwargs)
            for poison_cluster_class in poison_cluster_classes:
                result[idx[kwargs['cluster_class'] == poison_cluster_class]] = True
        return result

    def analyze_by_size(self, cluster_class: torch.Tensor, **kwargs) -> list[int]:
        r"""The smallest cluster.

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(1)``
        """
        return [cluster_class.bincount(minlength=self.nb_clusters).argmin().item()]

    def analyze_by_relative_size(self, cluster_class: torch.Tensor,
                                 size_threshold: float = 0.35,
                                 **kwargs) -> list[int]:
        r"""Small clusters whose proportion is smaller than :attr:`size_threshold`.

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.
            size_threshold (float): Defaults to ``0.35``.

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(K)``
        """
        relative_size = cluster_class.bincount(minlength=self.nb_clusters) / len(cluster_class)
        return torch.arange(self.nb_clusters)[relative_size < size_threshold].tolist()

    def analyze_by_silhouette_score(self, cluster_class: torch.Tensor,
                                    reduced_fm: torch.Tensor,
                                    silhouette_threshold: float = 0.1,
                                    **kwargs) -> list[int]:
        """Return :meth:`analyze_by_relative_size()`
        if :any:`sklearn.metrics.silhouette_score` is high,
        which means clustering fits data well.

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.
            reduced_fm (torch.Tensor): Dim-reduced feature map tensor
                with shape ``(N, self.nb_dims)``
            silhouette_threshold (float): The threshold to calculate
                :any:`sklearn.metrics.silhouette_score`.
                Defaults to ``0.1``.

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(K)``

        """
        if silhouette_score(reduced_fm, cluster_class) > silhouette_threshold:
            return self.analyze_by_relative_size(cluster_class, **kwargs)
        return []

    def analyze_by_distance(self, cluster_class: torch.Tensor,
                            reduced_fm: torch.Tensor,
                            reduced_fm_centers: torch.Tensor,
                            _class: int,
                            **kwargs) -> list[int]:
        r"""

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.
            reduced_fm (torch.Tensor): Dim-reduced feature map tensor
                with shape ``(N, self.nb_dims)``
            reduced_fm_centers (torch.Tensor): The centers of dim-reduced feature map tensors in each class
                with shape ``(C, self.nb_dims)``

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(K)``
        """
        cluster_centers_list = []
        for _class in range(self.nb_clusters):
            cluster_centers_list.append(reduced_fm[cluster_class == _class].median(dim=0))
        cluster_centers = torch.stack(cluster_centers_list)  # (self.nb_clusters, self.nb_dims)
        # (self.nb_clusters, C, self.nb_dims)
        differences = cluster_centers.unsqueeze(1) - reduced_fm_centers.unsqueeze(0)
        distances: torch.Tensor = differences.norm(p=2, dim=2)  # (self.nb_clusters, C)
        return torch.arange(self.nb_clusters)[distances.argmin(dim=1) != _class].tolist()
