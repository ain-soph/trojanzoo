#!/usr/bin/env python3

from ...abstract import BackdoorDefense
from trojanvision.environ import env
from trojanzoo.utils.data import dataset_to_tensor

import torch
from torch.utils.data import TensorDataset
import argparse

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class SpectralSignature(BackdoorDefense):
    """
    Spectral Signature Defense is described in the paper 'Spectral Signatures in Backdoor Attacks'_ by Brandon Tran. The main idea is backdoor attack tends to leave behind a detectable trace in the spectrum of the covariance of a feature representation learned by the neural network, that is if the means of the two populations are sufficiently well-separated relative to the variance of the populations, the corrupted datapoints can be detected and removed using singular value decomposition.
    The intuition is that if the set of inputs with a given label consists of both clean examples as well as corrupted examples from a different label set, the backdoor from the latter set will provide a strong signal in this representation for classiﬁcation. As long as the signal is large in magnitude, we can detect it via singular value decomposition and remove the images that provide the signal.

    The authors have not posted `original source code`_.

    Args:
        preprocess_layer (str): the chosen layer used to extract feature representation. Default: features.
        poison_image_num (int): the number of sampled poison image to train the  initial model. Default: 50.
        clean_image_num (int): the number of sampled clean image to train the  initial model. Default: 500.
        epsilon (int): the number of examples to remove from each class. Default: 5.
        retrain_epoch (int): the epochs to retrain the model on clean image dataset. Default: 10.

    .. _Spectral_Signature:
        https://arxiv.org/abs/1811.00636


    Returns:
        model(ImageModel): the clean model trained only with clean samples.
    """  # noqa: E501
    name: str = 'spectral_signature'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--preprocess_layer',
                           help='the chosen layer used to extract feature representation, defaults to ``flatten``')
        group.add_argument('--poison_image_num', type=int,
                           help='the number of sampled poison image to train the model initially, defaults to 50')
        group.add_argument('--clean_image_num', type=int,
                           help='the number of sampled clean image to train the model initially, defaults to 500')
        group.add_argument('--epsilon', type=int,
                           help='the number of examples to remove from each class, defaults to 5')
        group.add_argument('--retrain_epoch', type=int,
                           help='the epochs to retrain the model on clean image dataset, defaults to 5')
        return group

    def __init__(self, poison_image_num: int = 50, clean_image_num: int = 500,
                 preprocess_layer: str = 'flatten',
                 epsilon: int = 5,
                 retrain_epoch: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.preprocess_layer: str = preprocess_layer
        self.poison_image_num: int = poison_image_num
        self.clean_image_num: int = clean_image_num
        self.epsilon: int = epsilon
        self.retrain_epoch: int = retrain_epoch
        self.param_list['spectral_signature'] = ['preprocess_layer', 'poison_image_num', 'clean_image_num',
                                                 ' epsilon', 'retrain_epoch']

        clean_set, remain_dataset = self.dataset.split_dataset(
            dataset=self.dataset.get_dataset(mode='train'), length=self.clean_image_num)
        clean_input, clean_label = dataset_to_tensor(clean_set)
        self.clean_set = TensorDataset(clean_input, clean_label)

        poison_set, _ = self.dataset.split_dataset(dataset=remain_dataset, length=self.poison_image_num)
        trigger_input, trigger_label = dataset_to_tensor(poison_set)
        self.poison_set = TensorDataset(trigger_input, trigger_label)

        self.mix_dataset = torch.utils.data.ConcatDataset([self.clean_set, self.poison_set])
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
        initial_model._train(epochs=self.retrain_epoch, optimizer=optimizer,
                             lr_scheduler=lr_scheduler, loader_train=final_loader)

    def get_clean_dataloader(self):
        """
        Get the feature representation of samples from each class individually.
        Compute centered  representation.
        Singular value decomposition.
        Compute the vector of the outlier scores.
        Remove the examples with the top epsilon scores from the samples of each class and form the clean dataset.

        Returns:
            torch.utils.data.DataLoader:
                after removing the suspicious samples with bigger singular value,
                return the clean dataloader.
        """
        final_set = None    # TODO
        for k in range(self.dataset.num_classes):
            # class_dataset = self.dataset.get_class_subset(self.mix_dataset,classes = [k])
            idx = []
            for i, data in enumerate(self.mix_dataset):
                _input, _label = self.model.get_data(data)
                _input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                if _label.item() == k:
                    idx.append(k)
            class_dataset = torch.utils.data.Subset(self.mix_dataset, idx)
            class_input, class_label = dataset_to_tensor(class_dataset)
            class_dataset = TensorDataset(class_input, class_label)
            class_dataloader = self.dataset.get_dataloader(mode='train', dataset=class_dataset, num_workers=0)

            layer_output_all = []   # TODO
            for i, data in enumerate(class_dataloader):
                _input, _label = self.model.get_data(data)
                layer_output = self.model.get_layer(_input, layer_output=self.preprocess_layer)
                layer_output_all.append(layer_output.flatten(1))
            layer_output_all = torch.cat(layer_output_all, dim=0)
            layer_output_mean = torch.mean(layer_output_all, dim=0)

            for i in range(len(class_dataset)):
                layer_output_all[i] = layer_output_all[i] - layer_output_mean

            u, s, v = torch.svd(layer_output_all)
            vt = v.transpose(0, 1)
            outlier_scores = torch.rand([layer_output_all.shape[0]], device=env['device'])
            for i in range(len(class_dataset)):
                outlier_scores[i] = (layer_output_all[i].view(1, -1) @ vt[i].view(-1, 1)) ** 2
            outlier_scores_sorted, indices = torch.sort(outlier_scores, descending=True)

            clean_indices = indices[self.epsilon:]
            class_dataset = torch.utils.data.Subset(class_dataset, clean_indices)
            if k == 0:
                final_set = class_dataset
            else:
                final_set = torch.utils.data.ConcatDataset([final_set, class_dataset])

        final_dataloader = self.dataset.get_dataloader(mode=None, dataset=final_set, num_workers=0, pin_memory=False)
        return final_dataloader
