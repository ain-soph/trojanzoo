#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanvision.environ import env
from trojanzoo.utils.data import TensorListDataset

import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score
from sklearn import metrics
import argparse
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class ActivationClustering(BackdoorDefense):

    """
    Activation Clustering Defense is described in the paper 'Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering'_ by Bryant Chen. The main idea is the reason why backdoor and target samples receive the same classification is different, the network identifies features in the input taht it has learned corresponding to the target class, in the case of backdoor samples, it identifies features associated with the source class and the backdoor trigger. This difference in mechanism is evident in the network activations.

    The authors have posted `original source code`_. This defence is intergrated into the adversarial-robustness-toolbox provided by IBM.

    Args:
        mix_image_num (int): the number of sampled image, including clean image and poison image. Default: 50.
        clean_image_ratio (float): the ratio of clean image, compared to mix_image_num. Default: 0.5.
        retrain_epoch (int): the epoch of retraining the model, only with the detected clean image. Default: 10.
        nb_clusters (int): the amount of clusters. Default: 2. 
        clustering_method (str): the method for clustering the data, only support KMeans. Default: 'KMeans'.
        nb_dims (int): the dimension set in the process of reduceing the dimensionality of data. Default: 10.
        reduce_method (str): the method for reduing the dimensionality of data, only supporting ICA and PCA. Default: FastICA.
        cluster_analysis (str): the method chosen to analyze whether cluster is the poison cluster, including size, relative-size,exclusionary-reclassification, silhouette-scores. Default: size.



    .. _Activation Clustering:
        https://arxiv.org/abs/1811.03728

    .. _original source code:
        https://github.com/IBM/adversarial-robustness-toolbox


    Raises:
        ValueError: dimensionality reduction method not supported.
        ValueError: clustering method not supported.
        ValueError: *** Analyzer does not support more than two clusters.
        ValueError: Unsupported cluster analysis technique.
        ValueError: SC score too low, the sample should be contained in one cluster.

    Returns:
        model(ImageModel): the clean model trained only with clean samples.
    """

    name: str = 'activation_clustering'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--mix_image_num', type=int, help='the number of sampled image')
        group.add_argument('--clean_image_ratio', type=float, help='the ratio of clean image')
        group.add_argument('--retrain_epoch', type=int, help='the epoch of retraining the model')
        group.add_argument('--nb_clusters', type=int, help='')
        group.add_argument('--clustering_method', help='the amount of clusters')
        group.add_argument('--nb_dims', type=int,
                           help='the dimension set in the process of reduceing the dimensionality of data')
        group.add_argument('--reduce_method', help=' the method for reduing the dimensionality of data')
        group.add_argument('--cluster_analysis', help='the method chosen to analyze whether cluster is the poison cluster, '
                           'including size, distance, relative-size, silhouette-scores')
        return group

    def __init__(self, mix_image_num: int = 100, clean_image_ratio: float = 0.95, retrain_epoch: int = 10, nb_clusters: int = 2, clustering_method: str = "KMeans", nb_dims: int = 10, reduce_method: str = "FastICA", cluster_analysis: str = "exclusionary-reclassification", **kwargs):
        super().__init__(**kwargs)

        self.mix_image_num = mix_image_num
        self.clean_image_ratio = clean_image_ratio
        self.clean_image_num = int(mix_image_num * clean_image_ratio)
        self.poison_image_num = self.mix_image_num - self.clean_image_num

        self.nb_clusters = nb_clusters
        self.clustering_method = clustering_method
        self.nb_dims = nb_dims
        self.reduce_method = reduce_method
        self.cluster_analysis = cluster_analysis

        self.retrain_epoch = retrain_epoch

        self.clean_dataset, _ = self.dataset.split_dataset(
            self.dataset.get_full_dataset(mode='train'), self.clean_image_num)
        # clean_dataset, _ = self.dataset.split_dataset(self.dataset.get_full_dataset(mode='train'), self.clean_image_num)
        # clean_dataloader = self.dataset.get_dataloader(mode='train', dataset=clean_dataset, batch_size=self.clean_image_num, num_workers=0)
        # clean_imgs, _ = self.model.get_data(next(iter(clean_dataloader)))
        # self.clean_dataset = TensorDataset(clean_imgs, _)

        poison_dataset, _ = self.dataset.split_dataset(
            self.dataset.get_full_dataset(mode='train'), self.poison_image_num)
        poison_dataloader = self.dataset.get_dataloader(
            mode='train', dataset=poison_dataset, batch_size=self.poison_image_num, num_workers=0)
        poison_imgs, _ = self.model.get_data(next(iter(poison_dataloader)))
        poison_imgs = self.attack.add_mark(poison_imgs).cpu()
        poison_label = [self.attack.target_class] * self.poison_image_num
        self.poison_dataset = TensorListDataset(poison_imgs, poison_label)
        self.mix_dataset = torch.utils.data.ConcatDataset([self.clean_dataset, self.poison_dataset])
        self.mix_dataloader = self.dataset.get_dataloader(
            mode='train', dataset=self.mix_dataset, batch_size=self.dataset.batch_size, num_workers=0)

    def detect(self, optimizer, lr_scheduler, **kwargs):
        """
        Record the detected poison samples
        Remove them and retrain the model from scratch to get a clean model.
        """
        super().detect(**kwargs)
        all_reduced_activations, all_label, all_clusters, all_class = self.preprocess(self.mix_dataloader)
        poison_cluster_index = self.analyze_clusters(
            all_clusters, all_reduced_activations, all_label, all_class, self.cluster_analysis)
        print(all_class)
        poison_input_index = []
        for i in range(len(all_clusters)):
            if all_clusters[i] == poison_cluster_index:
                poison_input_index.append(i)

        y_pred = torch.zeros(self.mix_image_num)
        for i in range(len(poison_input_index)):
            y_pred[poison_input_index[i]] = 1

        y_true = torch.zeros(self.mix_image_num)
        for i in range(self.clean_image_num, self.mix_image_num):
            y_true[i] = 1

        # print("y_pred: ", y_pred)
        # print("y_true: ", y_true)

        final_f1_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score:", final_f1_score)
        print("precision_score:", metrics.precision_score(y_true, y_pred, average='weighted'))
        print("recall_score:", metrics.recall_score(y_true, y_pred, average='weighted'))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))

    def preprocess(self, loader):
        """
        Get the feature map of samples, convert to 1D tensor, reduce the dimensionality and cluster them.

        Args:
            loader (torch.utils.data.dataloader): the mix dataloader of clean image and poison image

        Returns:
            all_label (torch.Tensor): the prediction label of the model on all_input 
            all_reduced_activation (torch.FloatTensor): the reduced activation of all input getting from the model
            all_clusters (torch.Tensor): the clustering result
        """

        all_feature_map = []
        all_label = []
        all_class = []
        if env['tqdm']:
            loader = tqdm(loader)
        for i, data in enumerate(loader):
            _input, _label = self.model.get_data(data)
            feature_map = self.model._model.get_fm(_input)
            pred_label = self.model.get_class(_input)
            all_feature_map.append(feature_map)
            all_label.append(pred_label)
            all_class.append(_label)
        all_feature_map = torch.cat(all_feature_map)
        all_label = torch.cat(all_label)
        all_class = torch.cat(all_class)
        all_reduced_activations = self.reduce_dimensionality(all_feature_map, self.nb_dims, self.reduce_method)
        all_clusters = self.cluster_activations(all_reduced_activations, self.nb_clusters, self.clustering_method)
        all_clusters = torch.Tensor(all_clusters)
        return all_reduced_activations, all_label, all_clusters, all_class

    def reduce_dimensionality(self, activations, nb_dims: int = 10, reduce_method: str = "FastICA"):
        """
        Reduce dimensionality of activations.

        Args:
            activations (torch.FloatTensor): [description]
            nb_dims (int, optional): the setted dimensionality after reducing dimensionality. Defaults to 10.
            reduce_method (str, optional): the chosen reducing dimensionality method. Defaults to "FastICA".

        Raises:
            ValueError: dimensionality reduction method not supported.

        Returns:
            reduced_activations(torch.FloatTensor): the result after reduing dimensionality.
        """
        if reduce_method == "FastICA":
            projector = FastICA(n_components=nb_dims)
        elif reduce_method == "PCA":
            projector = PCA(n_components=nb_dims)
        else:
            raise ValueError(reduce_method + " dimensionality reduction method not supported.")
        activations = activations.view(activations.shape[0], -1)
        reduced_activations = projector.fit_transform(activations.detach().cpu())
        return reduced_activations

    def cluster_activations(self, reduced_activations, nb_clusters: int = 2, clustering_method: str = "KMeans"):
        """
        Cluster the activations after reducing dimensionality.

        Args:
            reduced_activations (torch.FloatTensor): the result after reduing dimensionality.
            nb_clusters (int, optional): the amount of clusters. Defaults to 2.
            clustering_method (str, optional): the chosen clustering method. Defaults to "KMeans".

        Raises:
            ValueError: clustering method not supported.

        Returns:
            clusters(torch.Tensor): the result of clustering
        """
        if clustering_method == "KMeans":
            clusterer = KMeans(n_clusters=nb_clusters)
            clusters = clusterer.fit_predict(reduced_activations)
            return clusters
        else:
            raise ValueError(clustering_method + " clustering method not supported.")

    def analyze_by_size(self, cluster_pred):
        """
        Analyze the result of clustering to judge which cluster is poison, according the size of clusters, usually the poisoned sample is less compared with normal data, so the smaller cluster will be regareded as poisoned samples.

        Args:
            cluster_pred (torch.Tensor): the result of clustering

        Raises:
            ValueError: Size Analyzer does not support more than two clusters.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(cluster_pred)) > 2):
            raise ValueError("Size Analyzer does not support more than two clusters.")
        num_1 = 0
        num_1 = torch.sum(cluster_pred)
        if num_1 > len(cluster_pred) / 2:
            poison_cluster_index = 0
            return poison_cluster_index
        else:
            poison_cluster_index = 1
            return poison_cluster_index

    def analyze_by_relative_size(self, label, cluster_pred, size_threshold: float = 0.35):
        """
        Analyze the result of clustering to judge which cluster is poison, according the ith label cluster result, if the portion of the smaller cluster is lower than size_threshold, this cluster is regarded as poison.

        Args:
            label (torch.Tensor): the original label of data in mix_dataloader.
            cluster_pred (torch.Tensor): the result of clustering
            size_threshold (float, optional): experience value. Defaults to 0.35.

        Raises:
            ValueError: Relative_Size Analyzer does not support more than two clusters.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(cluster_pred)) > 2):
            raise ValueError("Relative_Size Analyzer does not support more than two clusters.")
        else:
            all_classes = list(range(self.dataset.num_classes))
            for i in range(len(all_classes)):
                cur_label = all_classes[i]
                cur_label_num = torch.bincount(label)[cur_label]
                zip_label = list(zip(label, cluster_pred))
                num_0_cur_label = 0
                for j in range(len(zip_label)):
                    if list(zip_label[j])[0] == cur_label and list(zip_label[j])[1] == 0:
                        num_0_cur_label += 1
                if float(num_0_cur_label / cur_label_num) < size_threshold:
                    poison_cluster_index = 0
                    return poison_cluster_index
                elif float((cur_label_num - num_0_cur_label) / cur_label_num) < size_threshold:
                    poison_cluster_index = 1
                    return poison_cluster_index
                else:
                    continue

    def analyze_by_exclusionary_reclassification(self, cluster_pred, label, all_class):
        """
        Analyze the result of clustering to judge which cluster is poison, according the class of most data and the number of data of which the predicted and true class is identical. 

        Args:
            cluster_pred (torch.Tensor): the result of clustering
            label (torch.Tensor): the original label of data in mix_dataloader.
            all_class (torch.Tensor): the true class of sampled data.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        zip_label = list(zip(cluster_pred, label, all_class))
        # print('true label:',label)
        # print('predicted class:',all_class)
        max_label = torch.argmax(torch.bincount(label))
        print("max_label:{}".format(max_label))
        class_0_correct_num = 0
        class_1_correct_num = 0
        class_0_max_num = 0
        class_1_max_num = 0

        for i in range(len(zip_label)):
            if list(zip_label[i])[0] == 0:
                if list(zip_label[i])[1] == max_label:

                    class_0_max_num += 1
                if list(zip_label[i])[1] == list(zip_label[i])[2]:

                    class_0_correct_num += 1
            elif list(zip_label[i])[0] == 1:
                if list(zip_label[i])[1] == max_label:

                    class_1_max_num += 1
                if list(zip_label[i])[1] == list(zip_label[i])[2]:

                    class_1_correct_num += 1
        # try:
        if float(class_0_correct_num / (class_0_max_num + 1)) > float(class_1_correct_num / (class_1_max_num + 1)):
            poison_cluster_index = 1
            return poison_cluster_index
        else:
            poison_cluster_index = 0
            return poison_cluster_index
        # except ZeroDivisionError:
        #     print("Error: you can't divide by 0!")
        # else:
        #     poison_cluster_index = self.analyze_by_size(cluster_pred)
        #     return poison_cluster_index

    def analyze_by_silhouette_score(self, reduced_activation, cluster_pred, label, score_threshold: float = 0.1):
        """
        Analyze the result of clustering to judge which cluster is poison, according the silhouette_score, which specifies whether the number of clusters fits well with the data, the higher, the better. Test the situation under nb_clusters set as 2 and different class of data, if the silhouette_score is high, the smaller cluster will be the poison cluster.

        Args:
            reduced_activation (torch.FloatTensor): the reduced activation of sample in mix_dataloader.
            cluster_pred (torch.Tensor): the result of clustering.
            score_threshold (float, optional): experience value. Defaults to 0.1.

        Raises:
            ValueError: Silhouette_score Analyzer does not support more than two clusters.
            ValueError: SC score too low, the sample should be contained in one cluster.
            ValueError: clustering method not supported.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(cluster_pred)) > 2):
            raise ValueError("Silhouette_score Analyzer does not support more than two clusters.")
        else:
            if self.clustering_method == "KMeans":
                all_classes = list(range(self.dataset.num_classes))
                zip_label = list(zip(label, cluster_pred, reduced_activation))
                for i in range(len(all_classes)):
                    cur_label = all_classes[i]
                    cur_label_activation = []
                    cur_label_cluster_pred = []
                    for j in range(len(zip_label)):
                        if list(zip_label[j])[0] == cur_label:
                            cur_label_cluster_pred.append(list(zip_label[j])[1])
                            cur_label_activation.append(list(zip_label[j])[2])

                    if len(cur_label_activation) != 0:

                        cur_label_activation = torch.tensor(cur_label_activation)
                        cur_label_cluster_pred = torch.tensor(cur_label_cluster_pred)
                        kmeans_model = KMeans(n_clusters=self.nb_clusters).fit(cur_label_activation)
                        sc_score = silhouette_score(cur_label_activation, kmeans_model.labels_, metric='euclidean')
                        if sc_score > score_threshold:
                            poison_cluster_index = self.analyze_by_size(cur_label_cluster_pred)
                            return poison_cluster_index
                        else:
                            continue
                    else:
                        continue

            else:
                raise ValueError(self.clustering_method + " clustering method not supported.")

    def analyze_clusters(self, cluster_pred, reduced_activation, label, all_class, cluster_analysis: str = 'size', **kwargs):
        """
        Chooose the method of analyzing the clusters.

        Args:
            mix_feature_map (torch.FloatTensor): the feature map of sample in mix_dataloader.
            cluster_pred (torch.Tensor): the result of clustering.
            label (torch.Tensor): the original label of data in mix_dataloader.
            all_class (torch.Tensor): the true class of sampled data.
            cluster_analysis (str): the chosen cluster analyzing method.

        Returns:
            poison_cluster_index: the poisoned cluster number.

        """

        if cluster_analysis == "size":
            poison_cluster_index = self.analyze_by_size(cluster_pred)
        elif cluster_analysis == "relative-size":
            poison_cluster_index = self.analyze_by_relative_size(label, cluster_pred)
        elif cluster_analysis == "silhouette-scores":
            poison_cluster_index = self.analyze_by_silhouette_score(reduced_activation, cluster_pred, label)
        elif cluster_analysis == "exclusionary-reclassification":
            poison_cluster_index = self.analyze_by_exclusionary_reclassification(cluster_pred, label, all_class)
        else:
            raise ValueError("Unsupported cluster analysis technique " + cluster_analysis)
        return poison_cluster_index
