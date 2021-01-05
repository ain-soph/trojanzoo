#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..backdoor_defense import BackdoorDefense
from trojanvision.environ import env

import torch
import torch.utils.data
from torch.utils.data import TensorDataset
import argparse


class SpectralSignature(BackdoorDefense):

    """
    Spectral Signature Defense is described in the paper 'Spectral Signatures in Backdoor Attacks'_ by Brandon Tran. The main idea is backdoor attack tends to leave behind a detectable trace in the spectrum of the covariance of a feature representation learned by the neural network, that is if the means of the two populations are sufﬁciently well-separated relative to the variance of the populations, the corrupted datapoints can be detected and removed using singular value decomposition. 
    The intuition is that if the set of inputs with a given label consists of both clean examples as well as corrupted examples from a different label set, the backdoor from the latter set will provide a strong signal in this representation for classiﬁcation. As long as the signal is large in magnitude, we can detect it via singular value decomposition and remove the images that provide the signal. 

    The authors have not posted `original source code`_. 

    Args:
        preprocess_layer (str): the chosen layer used to extract feature representation. Default: features.
        poison_image_num (int): the number of sampled poison image to train the  initial model. Default: 50.
        clean_image_num (int): the number of sampled clean image to train the  initial model. Default: 500.
        epsilon (int): the number of examples to remove from each class. Default: 5.
        retrain_epoch (int): the epoch to retrain the model on clean image dataset. Default: 10.

    .. _Spectral_Signature:
        https://arxiv.org/abs/1811.00636


    Returns:
        model(ImageModel): the clean model trained only with clean samples.
    """
    name: str = 'spectral_signature'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                           help='the chosen layer used to extract feature representation, defaults to ``flatten``')
        group.add_argument('--poison_image_num', dest='poison_image_num', type=int,
                           help='the number of sampled poison image to train the model initially, defaults to 50')
        group.add_argument('--clean_image_num', dest='clean_image_num', type=int,
                           help='the number of sampled clean image to train the model initially, defaults to 500')
        group.add_argument('--epsilon', dest='epsilon', type=int,
                           help='the number of examples to remove from each class, defaults to 5')
        group.add_argument('--retrain_epoch', dest='retrain_epoch', type=int,
                           help='the epoch to retrain the model on clean image dataset, defaults to 5')

    def __init__(self, poison_image_num: int = 50, clean_image_num: int = 500, preprocess_layer: str = 'features', epsilon: int = 5, retrain_epoch: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.preprocess_layer: str = preprocess_layer
        self.poison_image_num: int = poison_image_num
        self.clean_image_num: int = clean_image_num
        self.epsilon: int = epsilon
        self.retrain_epoch: int = retrain_epoch

        self.clean_dataset, _ = self.dataset.split_set(
            dataset=self.dataset.get_full_dataset(mode='train'), length=self.clean_image_num)
        label_all = torch.empty([])    # TODO
        clean_input_all = torch.empty([])    # TODO
        for i, data in enumerate(iter(self.clean_dataset)):
            _input, _label = self.model.get_data(data)
            clean_input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
            if i == 0:
                clean_input_all = clean_input
                label_all = torch.unsqueeze(_label, 0)
            else:
                clean_input_all = torch.cat((clean_input_all, clean_input))
                label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
        label_all = torch.squeeze(label_all, 0)
        self.clean_dataset = TensorDataset(clean_input_all, label_all)
        self.clean_dataloader = self.dataset.get_dataloader(mode='train', dataset=self.clean_dataset, num_workers=0)

        self.poison_dataset, _ = self.dataset.split_set(dataset=_, length=self.poison_image_num)
        label_all = torch.empty([])    # TODO
        poison_input_all = torch.empty([])    # TODO
        for i, data in enumerate(iter(self.poison_dataset)):
            _input, _label = self.model.get_data(data)
            poison_input = self.attack.add_mark(_input)
            poison_input = poison_input.view(1, poison_input.shape[0], poison_input.shape[1], poison_input.shape[2])
            if i == 0:
                poison_input_all = poison_input
                label_all = torch.unsqueeze(_label, 0)
            else:
                poison_input_all = torch.cat((poison_input_all, poison_input))
                label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
        label_all = torch.squeeze(label_all, 0)
        self.poison_dataset = TensorDataset(poison_input_all, label_all)
        self.poison_dataloader = self.dataset.get_dataloader(
            mode='train', dataset=self.poison_dataset, num_workers=0, pin_memory=False)

        self.mix_dataset = torch.utils.data.ConcatDataset([self.clean_dataset, self.poison_dataset])
        self.mix_dataloader = self.dataset.get_dataloader(
            mode='train', dataset=self.mix_dataset, num_workers=0, pin_memory=False)

    def detect(self, optimizer, lr_scheduler, **kwargs):
        """
        Record the detected poison samples, remove them and retrain the model from scratch to get a clean model.
        """
        super().detect(**kwargs)
        initial_model = self.model
        self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, loader_train=self.mix_dataloader, **kwargs)
        final_loader = self.get_clean_dataloader()
        initial_model._train(epoch=self.retrain_epoch, optimizer=optimizer,
                             lr_scheduler=lr_scheduler, loader_train=final_loader)

    def get_clean_dataloader(self):
        """
        Get the feature representation of samples from each class individually.
        Compute centered  representation.
        Singular value decomposition.
        Compute the vector of the outlier scores.
        Remove the examples with the top epsilon scores from the samples of each class and form the clean dataset.

        Returns:
            torch.utils.data.DataLoader: after removing the suspicious samples with bigger singular value, return the clean dataloader.
        """
        final_set = None    # TODO
        for k in range(self.dataset.num_classes):
            # self.class_dataset = self.dataset.get_class_set(self.mix_dataset,classes = [k])
            idx = []
            for i, data in enumerate(self.mix_dataset):
                _input, _label = self.model.get_data(data)
                _input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                if _label.item() == k:
                    idx.append(k)
            self.class_dataset = torch.utils.data.Subset(self.mix_dataset, idx)

            layer_output_all = torch.empty([])    # TODO
            for i, data in enumerate(self.class_dataset):
                _input, _label = self.model.get_data(data)
                layer_output = self.model.get_layer(_input, layer_output=self.preprocess_layer)
                layer_output = layer_output.view(1, -1)
                if i == 0:
                    layer_output_all = layer_output
                else:
                    layer_output_all = torch.cat((layer_output_all, layer_output))

            layer_output_mean = torch.mean(layer_output_all, dim=0)

            for i in range(len(self.class_dataset)):
                layer_output_all[i] = layer_output_all[i] - layer_output_mean

            u, s, v = torch.svd(layer_output_all)
            v_transpose = torch.transpose(v, 1, 0)
            outlier_scores = torch.rand([layer_output_all.shape[0]], device=env['device'])
            for i in range(len(self.class_dataset)):
                outlier_scores[i] = (layer_output_all[i].view(1, -1) @ v_transpose[i].view(-1, 1)) ** 2
            outlier_scores_sorted, indices = torch.sort(outlier_scores, descending=True)

            clean_indices = indices[self.epsilon:]
            self.class_dataset = torch.utils.data.Subset(self.class_dataset, clean_indices)
            if k == 0:
                final_set = self.class_dataset
            else:
                final_set = torch.utils.data.ConcatDataset([final_set, self.class_dataset])

        final_dataloader = self.dataset.get_dataloader(mode=None, dataset=final_set, num_workers=0, pin_memory=False)
        return final_dataloader
