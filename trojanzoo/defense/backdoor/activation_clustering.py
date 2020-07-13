from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.process import Process
from ..defense_backdoor import Defense_Backdoor

from trojanzoo.utils import to_list
from trojanzoo.utils.model import AverageMeter
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.optim.uname import Uname
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import datetime
from tqdm import tqdm
from typing import List
from torch.autograd import Variable
import cv2
import sys
import math
import argparse
from operator import itemgetter
from heapq import nsmallest

from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from trojanzoo.utils import Config
env = Config.env


class Activation_Clustering(Defense_Backdoor):

    """
    Activation Clustering Defense is described in the paper 'Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering'_ by Bryant Chen. The main idea is the reason why backdoor and target samples receive the same classification is different, the network identifies features in the input taht it has learned corresponding to the target class, in the case of backdoor samples, it identifies features associated with the source class and the backdoor trigger. This difference in mechanism is evident in the network activations.

    The authors have posted `original source code`_. This defence is intergrated into the adversarial-robustness-toolbox provided by IBM.

    Args:
        dataset (ImageSet), model (ImageModel),optimizer (optim.Optimizer),lr_scheduler (optim.lr_scheduler._LRScheduler): the model, dataset, optimizer and lr_scheduler used in the whole procedure, specified in parser.
        mix_image_num (int): the number of sampled image, including clean image and poison image. Default: 50.
        clean_image_ratio (float): the ratio of clean image, compared to mix_image_num. Default: 0.5.
        retrain_epoch (int): the epoch of retraining the model, only with the detected clean image. Default: 10.
        nb_clusters (int): the amount of clusters. Default: 2. 
        clustering_method (str): the method for clustering the data, only support KMeans. Default: 'KMeans'.
        nb_dims (int): the dimension set in the process of reduceing the dimensionality of data. Default: 10.
        reduce_method (str): the method for reduing the dimensionality of data, only supporting ICA and PCA. Default: FastICA.
        cluster_analysis (str): the method chosen to analyze whether cluster is the poison cluster, including size, distance, relative-size, silhouette-scores.Default: size.



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

    def __init__(self, mix_image_num: int = 50, clean_image_ratio: float = 0.5, retrain_epoch: int = 10, nb_clusters: int = 2, clustering_method: str = "KMeans", nb_dims: int = 10, reduce_method: str = "FastICA", cluster_analysis: str = "size", **kwargs):
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

        # generate clean_dataloader, poison_dataloader
        self.clean_dataset, _ = self.dataset.split_set(dataset = self.dataset.get_full_dataset(mode='train'), length = self.clean_image_num)
        for i, data in enumerate(iter(self.clean_dataset)):
            _input, _label = self.model.get_data(data)
            clean_input = _input.view(1, _input.shape[0],_input.shape[1], _input.shape[2])
            if i == 0:
                clean_input_all = clean_input
                label_all = torch.unsqueeze(_label,0)
            else:
                clean_input_all = torch.cat((clean_input_all, clean_input))
                label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
        label_all = torch.squeeze(label_all, 0)
        self.clean_dataset = torch.utils.data.dataset.TensorDataset(clean_input_all, label_all)
        self.clean_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.clean_dataset, num_workers=0)


        self.poison_dataset,_ = self.dataset.split_set(dataset = _, length = self.poison_image_num)
        
        for i, data in enumerate(iter(self.poison_dataset)):
            _input, _label = self.model.get_data(data)
            poison_input  = self.attack.add_mark(_input)
            poison_input = poison_input.view(1, poison_input.shape[0],poison_input.shape[1], poison_input.shape[2])
            if i == 0:
                poison_input_all = poison_input
                label_all = torch.unsqueeze(_label,0)
            else:
                poison_input_all = torch.cat((poison_input_all, poison_input))
                label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
        label_all = torch.squeeze(label_all, 0)

        self.poison_dataset = torch.utils.data.dataset.TensorDataset(poison_input_all, label_all)
        self.poison_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.poison_dataset, num_workers=0, pin_memory=False)
        self.mix_dataset = torch.utils.data.ConcatDataset([self.clean_dataset,self.poison_dataset])
        self.mix_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.mix_dataset, num_workers=0, pin_memory=False)

            
    
    def detect(self, optimizer, lr_scheduler, **kwargs):
        """
        Record the detected poison samples, remove them and retrain the model from scratch to get a clean model.
        """
        super().detect(**kwargs)
        original_model, model, all_input, all_label, all_pred_label, all_feature_map, all_clusters = self.preprocess(self.mix_dataloader, optimizer, lr_scheduler, self.model) # to be modified
        reduced_activations = self.reduce_dimensionality(all_feature_map, self.nb_dims, self.reduce_method)
        clusters = self.cluster_activations(reduced_activations, self.nb_clusters, self.clustering_method)
        poison_cluster_index = self.analyze_clusters(all_clusters, all_feature_map, all_label, self.cluster_analysis)
        poison_input_index = []
        for i in range(len(all_clusters)):
            if all_clusters[i]==poison_cluster_index:
                poison_input_index.append(i)
        poison_input_index = torch.LongTensor(poison_input_index).to(all_feature_map.device)
        poison_input = torch.index_select(all_input, 0, poison_input_index)
        poison_label = torch.index_select(all_label, 0, poison_input_index)
        print(poison_input_index)
        for i in range(len(poison_input_index)):
            all_input = all_input[torch.arange(all_input.size(0))!=poison_input_index[i]]
            all_label = all_label[torch.arange(all_label.size(0))!=poison_input_index[i]]
        
        self.result_dataset = torch.utils.data.dataset.TensorDataset(all_input, all_label)
        self.result_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.result_dataset, num_workers=0, pin_memory=False)
        original_model._train(self.retrain_epoch, optimizer, lr_scheduler,loader_train=self.result_dataloader)

        
    
    def preprocess(self, loader, optimizer, lr_scheduler, model):
        """
        Get the feature map of samples, convert to 1D tensor, reduce the dimensionality and cluster them.

        Args:
            loader (torch.utils.data.dataloader): the mix dataloader of clean image and poison image
            model (Imagemodel): the initial model, assuming clean at the start

        Returns:
            original_model(Imagemodel): the initial model
            model(Imagemodel): the model after training on mix_dataloader
            all_input (torch.FloatTensor): all input in mix_dataloader
            all_label (torch.LongTensor): all label in mix_dataloader
            all_pred_label (torch.LongTensor): the prediction label of the model on all_input 
            all_feature_map (torch.FloatTensor): the feature map of all input getting from the model
            all_clusters (torch.LongTensor): the clustering result
        """
        
        original_model = model
        self.model._train(self.retrain_epoch, optimizer, lr_scheduler,loader_train=loader)
        
        for i, data in tqdm(enumerate(loader)):
            _input, _label = self.model.get_data(data)
            pred_label = self.model.get_class(_input)
            feature_map = self.model._model.get_fm(_input)
            if i == 0:
                all_input = _input
                all_label = _label
                all_pred_label = pred_label
                all_feature_map = feature_map
            else:
                all_input = torch.cat(all_input, _input)
                all_label = torch.cat(all_label, _label)
                all_pred_label = torch.cat(all_pred_label, pred_label)
                all_feature_map = torch.cat(all_feature_map, feature_map)
        
        all_reduced_activations = self.reduce_dimensionality(all_feature_map, self.nb_dims, self.reduce_method)
        all_clusters = self.cluster_activations(all_reduced_activations, self.nb_clusters, self.clustering_method)
        all_clusters = torch.LongTensor(all_clusters)
        return original_model, self.model, all_input, all_label, all_pred_label, all_feature_map, all_clusters
    

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
            raise ValueError(reduce_method  + " dimensionality reduction method not supported.")
        activations = activations.view(activations.shape[0],-1)
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
            clusters(torch.LongTensor): the result of clustering
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
            cluster_pred (torch.LongTensor): the result of clustering

        Raises:
            ValueError: Size Analyzer does not support more than two clusters.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(cluster_pred)) > 2):
            raise ValueError("Size Analyzer does not support more than two clusters.")
        num_1 = 0
        num_1 = torch.sum(cluster_pred)
        if num_1 > len(cluster_pred)/2:
            poison_cluster_index =0
            return  poison_cluster_index
        else:
            poison_cluster_index =1
            return  poison_cluster_index

    def analyze_by_distance(self, cluster_pred, mix_feature_map):
        """
        Analyze the result of clustering to judge which cluster is poison, according the distance to the center of corresponding cluster, usually the cluster of normal data is denser.

        Args:
            cluster_pred (torch.LongTensor): the result of clustering
            mix_feature_map (torch.FloatTensor): the feature map of sample in mix_dataloader.

        Raises:
            ValueError: Distance Analyzer does not support more than two clusters.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(cluster_pred)) > 2):
            raise ValueError("Distance Analyzer does not support more than two clusters.")
        else:
            zip_label_cluster = list(zip(cluster_pred, mix_feature_map))
            cluster_1_num = torch.sum(cluster_pred).item()
            cluster_0_num = len(zip_label_cluster) - cluster_1_num
            cluster_1_sample = torch.zeros(size =(cluster_1_num, mix_feature_map.shape[1]), device=env['device'])
            cluster_0_sample = torch.zeros(size =(cluster_0_num, mix_feature_map.shape[1]), device=env['device'])

            for i in range(len(zip_label_cluster)):
                if list(zip_label_cluster[i])[0] == 1:
                    cluster_1_sample[i] = mix_feature_map[i]
                else:
                    cluster_0_sample[i] = mix_feature_map[i]
            cluster_1_center = torch.median(cluster_1_sample, dim=0)
            cluster_0_center = torch.median(cluster_0_sample, dim=0)
            
            cluster_0_dist = 0
            for i in range(cluster_0_num):
                cluster_0_dist += torch.dist(cluster_0_center, cluster_0_sample[i])
            cluster_0_dist_average = cluster_0_dist/(cluster_0_num-1)

            cluster_1_dist = 0
            for i in range(cluster_1_num):
                cluster_1_dist += torch.dist(cluster_1_center, cluster_1_sample[i])
            cluster_1_dist_average = cluster_1_dist/(cluster_1_num-1)

            if cluster_0_dist_average > cluster_1_dist_average:
                poison_cluster_index = 0
                return  poison_cluster_index
            else:
                poison_cluster_index = 1
                return  poison_cluster_index
            

    def analyze_by_relative_size(self, label, cluster_pred, size_threshold: float = 0.35):
        """
        Analyze the result of clustering to judge which cluster is poison, according the relative number of constantly emerging class, usually the poisoned cluster contains more sample of this class.

        Args:
            label (torch.LongTensor): the original label of data in mix_dataloader.
            cluster_pred (torch.LongTensor): the result of clustering
            size_threshold (float, optional): experience value. Defaults to 0.35.

        Raises:
            ValueError: Relative_Size Analyzer does not support more than two clusters.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(cluster_pred))>2):
            raise ValueError("Relative_Size Analyzer does not support more than two clusters.")
        else:
            max_label = torch.argmax(torch.bincount(label))
            max_label_num = torch.bincount(label)[max_label]
            zip_label = list(zip(label, cluster_pred))
            num_0_max_label = 0
            for i in range(len(zip_label)):
                if list(zip_label[i])[0] == max_label and list(zip_label[i])[1] == 0:
                    num_0_max_label += 1
            if float(num_0_max_label / max_label_num) < size_threshold:
                poison_cluster_index = 0
                return  poison_cluster_index
            else:
                poison_cluster_index = 1
                return  poison_cluster_index


    def analyze_by_silhouette_score(self, mix_feature_map, cluster_pred, score_threshold: float = 0.1 ):
        """
        Analyze the result of clustering to judge which cluster is poison, according the silhouette_score, which specifies whether the number of clusters fits well with the data, the higher, the better. Test the situation under nb_clusters set as 2, if the silhouette_score is high, the smaller cluster will be the poison cluster.

        Args:
            mix_feature_map (torch.FloatTensor): the feature map of sample in mix_dataloader.
            cluster_pred (torch.LongTensor): the result of clustering.
            score_threshold (float, optional): experience value. Defaults to 0.1.

        Raises:
            ValueError: Silhouette_score Analyzer does not support more than two clusters.
            ValueError: SC score too low, the sample should be contained in one cluster.
            ValueError: clustering method not supported.

        Returns:
            poison_cluster_index : the poisoned cluster number.
        """
        if (len(torch.unique(y_pred))>2):
            raise ValueError("Silhouette_score Analyzer does not support more than two clusters.")
        else:
            if self.clustering_method == "KMeans":
                kmeans_model = KMeans(n_clusters=self.nb_clusters).fit(mix_feature_map)
                sc_score = silhouette_score(mix_feature_map, kmeans_model.labels_, metric='euclidean') 
                if sc_score > score_threshold:
                    poison_cluster_index = self.analyze_by_size(cluster_pred)
                    return poison_cluster_index
                else:
                    raise ValueError("SC score too low, the sample should be contained in one cluster.")     
            else:
                raise ValueError(self.clustering_method + " clustering method not supported.")

            
    def analyze_clusters(self, cluster_pred, mix_feature_map, label, cluster_analysis: str = 'size', **kwargs):
        """
        Chooose the method of analyzing the clusters.

        Args:
            mix_feature_map (torch.FloatTensor): the feature map of sample in mix_dataloader.
            cluster_pred (torch.LongTensor): the result of clustering.
            label (torch.LongTensor): the original label of data in mix_dataloader.
            cluster_analysis (str): the chosen cluster analyzing method.
        
        Returns:
            poison_cluster_index: the poisoned cluster number.

        """ 
        
        if cluster_analysis == "size":
            poison_cluster_index = self.analyze_by_size(cluster_pred)
        elif cluster_analysis == "distance":
            poison_cluster_index = self.analyze_by_distance(cluster_pred, mix_feature_map)
        elif cluster_analysis == "relative-size":
            poison_cluster_index = self.analyze_by_relative_size(label, cluster_pred)
        elif cluster_analysis == "silhouette-scores":
            poison_cluster_index = self.analyze_by_silhouette_score(mix_feature_map, cluster_pred)
        else:
            raise ValueError("Unsupported cluster analysis technique " + cluster_analysis)      
        return  poison_cluster_index 

    