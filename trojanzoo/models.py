#!/usr/bin/env python3

from trojanzoo.configs import config
from trojanzoo.datasets import Dataset
from trojanzoo.environ import env
from trojanzoo.utils.fim import KFAC, EKFAC
from trojanzoo.utils.model import (get_all_layer, get_layer, get_layer_name,
                                   activate_params, accuracy, generate_target,
                                   summary)
from trojanzoo.utils.module import get_name, BasicObject
from trojanzoo.utils.output import ansi, prints
from trojanzoo.utils.tensor import add_noise
from trojanzoo.utils.train import train, validate, compare

import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict
from collections.abc import Iterable

from typing import TYPE_CHECKING
# TODO: python 3.10
from typing import Generator, Iterator, Mapping
from trojanzoo.configs import Config    # TODO: python 3.10
from trojanzoo.utils.model import ExponentialMovingAverage
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import argparse
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data

__all__ = ['_Model', 'Model',
           'add_argument', 'create',
           'output_available_models']


class _Model(nn.Module):
    r"""A specific model class which inherits :any:`torch.nn.Module`.

    Args:
        num_classes (int): Number of classes.
        **kwargs: Keyword arguments passed to :meth:`define_preprocess`,
            :meth:`define_features` and :meth:`define_classifier`.

    Attributes:
        num_classes (int): Number of classes. Defaults to ``None``.

        preprocess (torch.nn.Module): Defaults to :meth:`define_preprocess()`.
        features (torch.nn.Module): Defaults to :meth:`define_features()`.
        pool (torch.nn.Module): :any:`torch.nn.AdaptiveAvgPool2d` ``((1, 1))``.
        classifier (torch.nn.Module): Defaults to :meth:`define_classifier()`.
    """

    def __init__(self, num_classes: int = None, **kwargs):
        super().__init__()
        self.preprocess = self.define_preprocess(**kwargs)
        self.features = self.define_features(**kwargs)   # feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # average pooling
        self.flatten = nn.Flatten()
        self.classifier = self.define_classifier(num_classes=num_classes, **kwargs)  # classifier

        self.num_classes = num_classes

    @classmethod
    def define_preprocess(self, **kwargs) -> nn.Module:
        r"""Define preprocess before feature extractor.

        Returns:
            torch.nn.Identity: Identity module.
        """
        return nn.Identity()

    @staticmethod
    def define_features(**kwargs) -> nn.Module:
        r"""Define feature extractor.

        Returns:
            torch.nn.Identity: Identity module.
        """
        return nn.Identity()

    @staticmethod
    def define_classifier(num_features: list[int] = [],
                          num_classes: int = 1000,
                          activation: type[nn.Module] = nn.ReLU,
                          activation_inplace: bool = True,
                          dropout: float = 0.0,
                          **kwargs) -> nn.Sequential:
        r"""
        | Define classifier as
            ``(Linear -> Activation -> Dropout ) * (len(num_features) - 1) -> Linear``.
        | If there is only 1 linear layer, its name will be ``'fc'``.
        | Else, all layer names will be indexed starting from ``0``
            (e.g., ``'fc1', 'relu1', 'dropout0'``).

        Args:
            num_features (list[int]): List of feature numbers.
                Each element serves as the :attr:`in_features` of current layer
                and :attr:`out_features` of preceding layer.
                Defaults to ``[]``.
            num_classes (int): The number of classes.
                This serves as the :attr:`out_features` of last layer.
                Defaults to ``None``.
            activation (type[torch.nn.Module]):
                The type of activation layer.
                Defaults to :any:`torch.nn.ReLU`.
            activation_inplace (bool): Whether to use inplace activation.
                Defaults to ``'True'``
            dropout (float): The drop out probability.
                Will **NOT** add dropout layers if it's ``0``.
                Defaults to ``0.0``.
            **kwargs: Any keyword argument (unused).

        Returns:
            torch.nn.Sequential: The sequential classifier.

        :Examples:
            >>> from trojanzoo.models import _Model
            >>>
            >>> _Model.define_classifier(num_features=[5,4,4], num_classes=10)
            Sequential(
                (fc1): Linear(in_features=5, out_features=4, bias=True)
                (relu1): ReLU(inplace=True)
                (dropout1): Dropout(p=0.5, inplace=False)
                (fc2): Linear(in_features=4, out_features=4, bias=True)
                (relu2): ReLU(inplace=True)
                (dropout2): Dropout(p=0.5, inplace=False)
                (fc3): Linear(in_features=4, out_features=10, bias=True)
            )
        """
        seq = nn.Sequential()
        if len(num_features) == 0:
            return seq
        if activation:
            activation_name = activation.__name__.split('.')[-1].lower()
        if len(num_features) == 1:
            seq.add_module('fc', nn.Linear(num_features[0], num_classes))
        else:
            for i in range(len(num_features) - 1):
                seq.add_module(
                    f'fc{i + 1:d}', nn.Linear(num_features[i], num_features[i + 1]))
                if activation:
                    seq.add_module(f'{activation_name}{i + 1:d}',
                                   activation(inplace=activation_inplace))
                if dropout > 0:
                    seq.add_module(f'dropout{i + 1:d}', nn.Dropout(p=dropout))
            seq.add_module(f'fc{len(num_features):d}', nn.Linear(num_features[-1], num_classes))
        return seq

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""``x -> self.get_final_fm -> self.classifier -> return``"""
        x = self.get_final_fm(x, **kwargs)
        x = self.classifier(x)
        return x

    # TODO: combine with get_final_fm ? Consider GNN cases.
    def get_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""``x -> self.preprocess -> self.features -> return``"""
        return self.features(self.preprocess(x))

    def get_final_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""``x -> self.get_fm -> self.pool -> self.flatten -> return``"""
        x = self.get_fm(x, **kwargs)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)


class Model(BasicObject):
    r"""
    | A general model wrapper class, which should be the most common interface for users.
    | It inherits :class:`trojanzoo.utils.module.BasicObject`.

    Args:
        name (str): Name of model.
        suffix (str):
            | Suffix of local model weights file (e.g., ``'_adv_train'``).
              Defaults to empty string ``''``.
            | The location of local pretrained weights is
              ``'{folder_path}/{self.name}{self.suffix}.pth'``
        model (type[_Model] | _Model): Type of model or a specific model instance.
        dataset (trojanzoo.datasets.Dataset | None): Corresponding dataset (optional).
            Defaults to ``None``.
        num_classes (int | None): Number of classes.
            If it's ``None``, fetch the value from :attr:`dataset`.
            Defaults to ``None``.
        folder_path (str): Folder path to save model weights.
            Defaults to ``None``.

            Note:
                :attr:`folder_path` is usually
                ``'{model_dir}/{dataset.data_type}/{dataset.name}'``,
                which is claimed as the default value of :func:`create()`.
        official (bool): Whether to use official pretrained weights.
            Defaults to ``False``.
        pretrained (bool): Whether to use local pretrained weights
            from ``'{folder_path}/{self.name}{self.suffix}.pth'``
            Defaults to ``False``.
        randomized_smooth (bool): Whether to use randomized smoothing.
            Defaults to ``False``.
        rs_sigma (float): Randomized smoothing sampling std :math:`\sigma`.
            Defaults to ``0.01``.
        rs_n (int): Randomized smoothing sampling number. Defaults to ``100``.

    Attributes:
        available_models (list[str]): The list of available model names.
        model_urls (dict[str, str]): The links to official pretrained model weights.

        name (str): Name of model.
        suffix (str):
            | Suffix of local model weights file (e.g., ``'_adv_train'``).
              Defaults to empty string ``''``.
            | The location of local pretrained weights is
              ``'{folder_path}/{self.name}{self.suffix}.pth'``
        _model (_Model): :any:`torch.nn.Module` model instance.
        model (torch.nn.DataParallel | _Model):
            Parallel version of :attr:`_model` if there is more than 1 GPU available.
            Generated by :meth:`get_parallel_model()`.
        dataset (trojanzoo.datasets.Dataset | None): Corresponding dataset (optional).
            Defaults to ``None``.
        num_classes (int | None): Number of classes.
            If it's ``None``, fetch the value from :attr:`dataset`.
            Defaults to ``None``.
        folder_path (str): Folder path to save model weights.
            Defaults to ``None``.
        randomized_smooth (bool): Whether to use randomized smoothing.
            Defaults to ``False``.
        rs_sigma (float): Randomized smoothing sampling std :math:`\sigma`.
        rs_n (int): Randomized smoothing sampling number. Defaults to ``100``.

        criterion (~collections.abc.Callable):
            The criterion used to calculate :meth:`loss()`.
        criterion_noreduction (~collections.abc.Callable):
            The criterion used to calculate :meth:`loss()`
            when ``reduction='none'``.
        softmax (torch.nn.Module): :any:`torch.nn.Softmax` ``(dim=1)``.
            Used in :meth:`get_prob()`.
    """

    available_models: list[str] = []
    model_urls: dict[str, str] = []

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
        r"""Add model arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete model class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.
        """
        group.add_argument('-m', '--model', dest='model_name',
                           help='model name '
                           '(default: config[model][default_model])')
        group.add_argument('--suffix',
                           help='model name suffix (e.g., "_adv_train")')
        group.add_argument('--pretrained', action='store_true',
                           help='load local pretrained weights (default: False)')
        group.add_argument('--official', action='store_true',
                           help='load official pretrained weights (default: False)')
        group.add_argument('--randomized_smooth',
                           help='whether to use randomized smoothing '
                           '(default: False)')
        group.add_argument('--rs_sigma', type=float,
                           help='randomized smoothing sampling std '
                           '(default: 0.01)')
        group.add_argument('--rs_n', type=int,
                           help='randomized smoothing sampling number '
                           '(default: 100)')
        group.add_argument('--model_dir', help='directory to store pretrained models')
        return group

    def __init__(self, name: str = 'model', suffix: str = None,
                 model: type[_Model] | _Model = _Model,
                 dataset: Dataset = None,
                 num_classes: int = None, folder_path: str = None,
                 official: bool = False, pretrained: bool = False,
                 randomized_smooth: bool = False,
                 rs_sigma: float = 0.01, rs_n: int = 100, **kwargs):
        super().__init__()
        self.param_list['model'] = ['folder_path']
        if suffix is not None:
            self.param_list['model'].append('suffix')
        else:
            suffix = ''
        if randomized_smooth:
            self.param_list['model'].extend(
                ['randomized_smooth', 'rs_sigma', 'rs_n'])
        self.name: str = name
        self.dataset = dataset
        self.suffix = suffix
        self.randomized_smooth: bool = randomized_smooth
        self.rs_sigma: float = rs_sigma
        self.rs_n: int = rs_n

        self.folder_path = folder_path
        if folder_path is not None:
            self.folder_path = os.path.normpath(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # ------------Auto-------------- #
        loss_weights: torch.Tensor = kwargs.get('loss_weights', None)
        if dataset:
            if not isinstance(dataset, Dataset):
                raise TypeError(f'{type(dataset)=}    {dataset=}')
            num_classes = num_classes or dataset.num_classes
            loss_weights = kwargs.get('loss_weights', dataset.loss_weights)
        self.num_classes = num_classes  # number of classes
        # TODO: what device shall we save loss_weights?
        # numpy, torch, or torch.cuda.

        match loss_weights:
            case np.ndarray():
                loss_weights = torch.from_numpy(loss_weights).to(device=env['device'], dtype=torch.float)
            case torch.Tensor():
                loss_weights = loss_weights.to(device=env['device'], dtype=torch.float)

        self.loss_weights = loss_weights
        self.layer_name_list: list[str] = None

        # ------------------------------ #
        self.criterion = self.define_criterion(weight=loss_weights)
        self.criterion_noreduction = self.define_criterion(
            weight=loss_weights, reduction='none')
        self.softmax = nn.Softmax(dim=1)
        match model:
            case type():
                if num_classes is not None:
                    kwargs['num_classes'] = num_classes
                self._model = model(name=name, dataset=dataset, **kwargs)
            case nn.Module():
                self._model = model
            case _:
                raise TypeError(type(model))
        self.model = self.get_parallel_model(self._model)
        self.activate_params([])
        if official:
            self.load('official')
        if pretrained:
            self.load(verbose=True)
        self.eval()
        if env['num_gpus']:
            self.cuda()

    # ----------------- Forward Operations ----------------------#

    def get_logits(self, _input: torch.Tensor, parallel: bool = False,
                   randomized_smooth: bool = None,
                   rs_sigma: float = None, rs_n: int = None,
                   **kwargs) -> torch.Tensor:
        r"""Get logits of :attr:`_input`.

        Note:
            Users should use model as Callable function
            rather than call this method directly,
            because ``__call__`` supports :any:`torch.cuda.amp`.

        Args:
            _input (torch.Tensor): The batched input tensor.
            parallel (bool): Whether to use parallel model
                `self.model` rather than `self._model`.
                Defautls to ``False``.
            randomized_smooth (bool | None): Whether to use randomized smoothing.
                If it's ``None``, use :attr:`self.randmized_smooth` instead.
                Defaults to ``None``.
            rs_sigma (float | None): Randomized smoothing sampling std :math:`\sigma`.
                If it's ``None``, use :attr:`self.rs_sigma` instead.
                Defaults to ``None``.
            rs_n (int): Randomized smoothing sampling number.
                If it's ``None``, use :attr:`self.rs_n` instead.
                Defaults to ``None``.
            **kwargs: Keyword arguments passed to :meth:`forward()`.

        Returns:
            torch.Tensor: The logit tensor with shape ``(N, C)``.
        """
        model = self.model if parallel else self._model
        if randomized_smooth is None:
            randomized_smooth = self.randomized_smooth
        if randomized_smooth:
            rs_sigma = rs_sigma if rs_sigma is not None else self.rs_sigma
            rs_n = rs_n if rs_n is not None else self.rs_n
            _list = []
            for _ in range(rs_n):
                # TODO: valid input clip issue
                _input_noise = add_noise(_input, std=rs_sigma)
                _list.append(model(_input_noise, **kwargs))
            return torch.stack(_list).mean(dim=0)
            # TODO: memory issues and parallel possibilities
            # _input_noise = add_noise(repeat_to_batch(
            #     _input, batch_size=n), std=sigma).flatten(end_dim=1)
            # return model(_input_noise, **kwargs).view(
            #     n, len(_input), self.num_classes).mean(dim=0)
        else:
            return model(_input, **kwargs)

    def get_fm(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the feature map of :attr:`_input`,
        which is the output of :attr:`self.features`
        and input of :attr:`self.pool`.
        Call :meth:`_Model.get_fm()`.

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_fm()`.
            **kwargs: Keyword arguments passed to :meth:`_Model.get_fm()`.

        Returns:
            torch.Tensor: The feature tensor with shape ``(N, C', H', W')``.
        """
        return self._model.get_fm(_input, **kwargs)

    def get_final_fm(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the final feature map of :attr:`_input`,
        which is the output of :attr:`self.flatten`
        and input of :attr:`self.classifier`.
        Call :meth:`_Model.get_final_fm()`.

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_final_fm()`.
            **kwargs: Keyword arguments passed to :meth:`_Model.get_final_fm()`.

        Returns:
            torch.Tensor: The feature tensor with shape ``(N, dim)``.
        """
        return self._model.get_final_fm(_input, **kwargs)

    def get_prob(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the probability classification vector of :attr:`_input`.

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_logits()`.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`.

        Returns:
            torch.Tensor: The probability tensor with shape ``(N, C)``.
        """
        return self.softmax(self(_input, **kwargs))

    def get_target_prob(self, _input: torch.Tensor,
                        target: int | list[int] | torch.Tensor,
                        **kwargs) -> torch.Tensor:
        r"""Get the probability w.r.t. :attr:`target` class of :attr:`_input`
        (using :any:`torch.gather`).

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_logits()`.
            target (int | list[int] | torch.Tensor): Batched target classes.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`.

        Returns:
            torch.Tensor: The probability tensor with shape ``(N)``.
        """
        match target:
            case int():
                target = [target] * len(_input)
            case list():
                target = torch.tensor(target, device=_input.device)
        return self.get_prob(_input, **kwargs).gather(
            dim=1, index=target.unsqueeze(1)).flatten()

    def get_class(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the class classification result of :attr:`_input`
        (using :any:`torch.argmax`).

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_logits()`.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`.

        Returns:
            torch.Tensor: The classes tensor with shape ``(N)``.
        """
        return self(_input, **kwargs).argmax(dim=-1)

    def get_layer_name(self, depth: int = -1, prefix: str = '',
                       use_filter: bool = True, non_leaf: bool = False,
                       seq_only: bool = False) -> list[str]:
        r"""Get layer names of model instance.

        Args:
            depth (int): The traverse depth.
                Defaults to ``-1`` (means :math:`\infty`).
            prefix (str): The prefix string to all elements.
                Defaults to empty string ``''``.
            use_filter (bool): Whether to filter out certain layer types.

                * :any:`torchvision.transforms.Normalize`
                * :any:`torch.nn.Dropout`
                * :any:`torch.nn.BatchNorm2d`
                * :any:`torch.nn.ReLU`
                * :any:`torch.nn.Sigmoid`
            non_leaf (bool): Whether to include non-leaf nodes.
                Defaults to ``False``.
            seq_only (bool): Whether to only traverse children
                of :any:`torch.nn.Sequential`.
                If ``False``, will traverse children of all :any:`torch.nn.Module`.
                Defaults to ``False``.

        Returns:
            list[str]: The list of all layer names.

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.model.get_layer_name()`.
        """
        return get_layer_name(self._model, depth, prefix,
                              use_filter, non_leaf, seq_only)

    def get_all_layer(self, _input: torch.Tensor,
                      layer_input: str = 'input', depth: int = -1,
                      prefix='', use_filter: bool = True, non_leaf: bool = False,
                      seq_only: bool = True, verbose: int = 0
                      ) -> dict[str, torch.Tensor]:
        r"""Get all intermediate layer outputs of
        :attr:`_input` from any intermediate layer.

        Args:
            _input (torch.Tensor): The batched input tensor
                from :attr:`layer_input`.
            layer_input (str): The intermediate layer name of :attr:`_input`.
                Defaults to ``'input'``.
            depth (int): The traverse depth.
                Defaults to ``-1`` (:math:`\infty`).
            prefix (str): The prefix string to all elements.
                Defaults to empty string ``''``.
            use_filter (bool): Whether to filter out certain layer types.

                * :any:`torchvision.transforms.Normalize`
                * :any:`torch.nn.Dropout`
                * :any:`torch.nn.BatchNorm2d`
                * :any:`torch.nn.ReLU`
                * :any:`torch.nn.Sigmoid`
            non_leaf (bool): Whether to include non-leaf nodes.
                Defaults to ``False``.
            seq_only (bool): Whether to only traverse children
                of :any:`torch.nn.Sequential`.
                If ``False``, will traverse children of all :any:`torch.nn.Module`.
                Defaults to ``False``.
            verbose (bool): The output level to show information
                including layer name, output shape and module information.
                Setting it larger than ``0`` will enable the output.
                Different integer values stands for different module information.
                Defaults to ``0``.

                * ``0``: No output
                * ``1``: Show layer class name.
                * ``2``: Show layer string (first line).
                * ``3``: Show layer string (full).

        Returns:
            dict[str, torch.Tensor]: The dict of all layer outputs.

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.model.get_all_layer()`.
        """
        return get_all_layer(self._model, _input, layer_input, depth,
                             prefix, use_filter, non_leaf, seq_only, verbose)

    def get_layer(self, _input: torch.Tensor, layer_output: str = 'classifier',
                  layer_input: str = 'input',
                  seq_only: bool = True) -> torch.Tensor:
        r"""Get one certain intermediate layer output
        of :attr:`_input` from any intermediate layer.

        Args:
            _input (torch.Tensor): The batched input tensor
                from :attr:`layer_input`.
            layer_output (str): The intermediate output layer name.
                Defaults to ``'classifier'``.
            layer_input (str): The intermediate layer name of :attr:`_input`.
                Defaults to ``'input'``.
            seq_only (bool): Whether to only traverse children
                of :any:`torch.nn.Sequential`.
                If ``False``, will traverse children of all :any:`torch.nn.Module`.
                Defaults to ``True``.

        Returns:
            torch.Tensor: The output of layer :attr:`layer_output`.

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.model.get_layer()`.
        """
        if layer_input == 'input':
            match layer_output:
                case 'classifier':
                    return self(_input)
                case 'features':
                    return self._model.get_fm(_input)
                case 'flatten':
                    return self.get_final_fm(_input)
        if self.layer_name_list is None:
            self.layer_name_list: list[str] = self.get_layer_name(
                use_filter=False, non_leaf=True)
            self.layer_name_list.insert(0, 'input')
            self.layer_name_list.append('output')
        return get_layer(self._model, _input, layer_output, layer_input,
                         layer_name_list=self.layer_name_list,
                         seq_only=seq_only)

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, reduction: str = 'mean',
             **kwargs) -> torch.Tensor:
        r"""Calculate the loss using :attr:`self.criterion`
        (:attr:`self.criterion_noreduction`).

        Args:
            _input (torch.Tensor | None): The batched input tensor.
                If :attr:`_output` is provided, this argument will be ignored.
                Defaults to ``None``.
            _label (torch.Tensor): The label of the batch with shape ``(N)``.
            _output (torch.Tensor | None): The logits of :attr:`_input`.
                If ``None``, use :attr:`_input` to calculate logits.
                Defaults to ``None``.
            reduction (str): Specifies the reduction to apply to the output.
                Choose from ``['none', 'mean']``.
                Defaults to ``'mean'``.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`
                if :attr:`_output` is not provided.

        Returns:
            torch.Tensor:
                A scalar loss tensor (with shape ``(N)`` if ``reduction='none'``).
        """
        criterion = self.criterion_noreduction if reduction == 'none' \
            else self.criterion
        if _output is None:
            _output = self(_input, **kwargs)
        return criterion(_output, _label)

    # -------------------------------------------------------- #

    def define_optimizer(
            self, parameters: str | Iterator[nn.Parameter] = 'full',
            OptimType: str | type[Optimizer] = 'SGD',
            lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0,
            lr_scheduler: bool = False,
            lr_scheduler_type: str = 'CosineAnnealingLR',
            lr_step_size: int = 30, lr_gamma: float = 0.1,
            epochs: int = None, lr_min: float = 0.0,
            lr_warmup_epochs: int = 0, lr_warmup_method: str = 'constant',
            lr_warmup_decay: float = 0.01,
            **kwargs) -> tuple[Optimizer, _LRScheduler]:
        r"""Define optimizer and lr_scheduler.

        Args:
            parameters (str | ~collections.abc.Iterable[torch.nn.parameter.Parameter]):
                The parameters to optimize while other model parameters are frozen.
                If :class:`str`, set :attr:`parameters` as:

                    * ``'features': self._model.features``
                    * ``'classifier' | 'partial': self._model.classifier``
                    * ``'full': self._model``

                Defaults to ``'full'``.
            OptimType (str | type[Optimizer]):
                The optimizer type.
                If :class:`str`, load from module :any:`torch.optim`.
                Defaults to ``'SGD'``.
            lr (float): The learning rate of optimizer. Defaults to ``0.1``.
            momentum (float): The momentum of optimizer. Defaults to ``0.0``.
            weight_decay (float): The momentum of optimizer. Defaults to ``0.0``.
            lr_scheduler (bool): Whether to enable lr_scheduler. Defaults to ``False``.
            lr_scheduler_type (str): The type of lr_scheduler.
                Defaults to ``'CosineAnnealingLR'``.

                Available lr_scheduler types (use string rather than type):

                    * :any:`torch.optim.lr_scheduler.StepLR`
                    * :any:`torch.optim.lr_scheduler.CosineAnnealingLR`
                    * :any:`torch.optim.lr_scheduler.ExponentialLR`
            lr_step_size (int): :attr:`step_size` for :any:`torch.optim.lr_scheduler.StepLR`.
                Defaults to ``30``.
            lr_gamma (float): :attr:`gamma` for :any:`torch.optim.lr_scheduler.StepLR`
                or :any:`torch.optim.lr_scheduler.ExponentialLR`.
                Defaults to ``0.1``.
            epochs (int): Total training epochs.
                ``epochs - lr_warmup_epochs`` is passed as :attr:`T_max`
                to any:`torch.optim.lr_scheduler.CosineAnnealingLR`.
                Defaults to ``None``.
            lr_min (float): The minimum of learning rate.
                It's passed as :attr:`eta_min`
                to any:`torch.optim.lr_scheduler.CosineAnnealingLR`.
                Defaults to ``0.0``.
            lr_warmup_epochs (int): Learning rate warmup epochs.
                Passed as :attr:`total_iters` to lr_scheduler.
                Defaults to ``0``.
            lr_warmup_method (str): Learning rate warmup methods.
                Choose from ``['constant', 'linear']``.
                Defaults to ``'constant'``.
            lr_warmup_decay (float): Learning rate warmup decay factor.
                Passed as :attr:`factor` (:attr:`start_factor`) to lr_scheduler.
                Defaults to ``0.01``.
            **kwargs: Keyword arguments passed to optimizer init method.

        Returns:
            (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler):
                The tuple of optimizer and lr_scheduler.
        """
        kwargs['momentum'] = momentum
        kwargs['weight_decay'] = weight_decay
        match parameters:
            case str():
                parameters = self.get_parameter_from_name(name=parameters)
            case Iterable():
                pass
            case _:
                raise TypeError(f'{type(parameters)=}    {parameters=}')
        if isinstance(OptimType, str):
            OptimType: type[Optimizer] = getattr(torch.optim, OptimType)
        keys = OptimType.__init__.__code__.co_varnames
        kwargs = {k: v for k, v in kwargs.items() if k in keys}
        optimizer = OptimType(parameters, lr, **kwargs)
        _lr_scheduler: _LRScheduler = None
        if lr_scheduler:
            main_lr_scheduler: _LRScheduler
            match lr_scheduler_type:
                case 'StepLR':
                    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=lr_step_size, gamma=lr_gamma)
                case 'CosineAnnealingLR':
                    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=epochs - lr_warmup_epochs, eta_min=lr_min)
                case 'ExponentialLR':
                    main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=lr_gamma)
                case _:
                    raise NotImplementedError(
                        f'Invalid {lr_scheduler_type=}.'
                        'Only "StepLR", "CosineAnnealingLR" and "ExponentialLR" '
                        'are supported.')
            if lr_warmup_epochs > 0:
                match lr_warmup_method:
                    case 'linear':
                        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=lr_warmup_decay,
                            total_iters=lr_warmup_epochs)
                    case 'constant':
                        warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                            optimizer, factor=lr_warmup_decay,
                            total_iters=lr_warmup_epochs)
                    case _:
                        raise NotImplementedError(
                            f'Invalid {lr_warmup_method=}.'
                            'Only "linear" and "constant" are supported.')
                _lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                    milestones=[lr_warmup_epochs])
            else:
                _lr_scheduler = main_lr_scheduler
        return optimizer, _lr_scheduler

    # TODO: linting, or maybe nn.Module for generic?
    def define_criterion(self, **kwargs) -> nn.CrossEntropyLoss:
        r"""Define criterion to calculate loss.
        Defaults to use :any:`torch.nn.CrossEntropyLoss`.

        Args:
            weight (torch.Tensor | None):
                The loss weights passed to :any:`torch.nn.CrossEntropyLoss`.
                Defaults to :attr:`self.loss_weights`.
            **kwargs: Keyword arguments passed to :any:`torch.nn.CrossEntropyLoss`.
        """
        if 'weight' not in kwargs.keys():
            kwargs['weight'] = self.loss_weights
        return nn.CrossEntropyLoss(**kwargs)
        # if loss_type == 'jsd':
        #     num_classes = num_classes if num_classes is not None \
        #       else self.num_classes

        #     def jsd(_output: torch.Tensor, _label: torch.Tensor, **kwargs):
        #         p: torch.Tensor = F.one_hot(_label, num_classes)
        #         q: torch.Tensor = F.softmax(_output)
        #         log_q = F.log_softmax(_output)
        #         sum_pq = p + q
        #         loss = sum_pq * (sum_pq.log() - math.log(2))
        # return criterion

    # ---------------------Load & Save Model------------------------- #

    @torch.no_grad()
    def load(self, file_path: str = None, folder_path: str = None,
             suffix: str = None, inplace: bool = True,
             map_location: str | Callable | torch.device | dict = 'cpu',
             component: str = 'full', strict: bool = True,
             verbose: bool = False, indent: int = 0,
             **kwargs) -> OrderedDict[str, torch.Tensor]:
        r"""Load pretrained model weights.

        Args:
            file_path (str | None): The file path to load pretrained weights.
                If ``'official'``, call :meth:`get_official_weights()`.
                Defaults to ``'{folder_path}/{self.name}{suffix}.pth'``.
            folder_path (str | None): The folder path containing model checkpoint.
                It is used when :attr:`file_path` is not provided.
                Defaults to :attr:`self.folder_path`.
            suffix (str | None): The suffix string to model weights file.
                Defaults to :attr:`self.suffix`.
            inplace (bool): Whether to change model parameters.
                If ``False``, will only return the dict but not change model parameters.
                Defaults to ``True``.
            map_location (str | ~torch.torch.device | dict):
                Passed to :any:`torch.load`.
                Defaults to ``'cpu'``.

                Note:
                    The device of model parameters will still be ``'cuda'``
                    if there is any cuda available.
                    This argument only affects intermediate operation.
            component (str): Specify which part of the weights to load.
                Choose from ``['full', 'features', 'classifier']``.
                Defaults to ``'full'``.
            strict (bool): Passed to :any:`torch.nn.Module.load_state_dict`.
                Defaults to ``True``.
            verbose (bool): Whether to output auxiliary information.
                Defaults to ``False``.
            indent (int): The indent of output auxialiary information.
            **kwargs: Keyword arguments passed to :any:`torch.load`.

        Returns:
            OrderedDict[str, torch.Tensor]: The model weights OrderedDict.
        """
        map_location = map_location if map_location != 'default' \
            else env['device']
        if file_path is None:
            folder_path = folder_path if folder_path is not None \
                else self.folder_path
            suffix = suffix if suffix is not None else self.suffix
            file_path = os.path.normpath(os.path.join(
                folder_path, f'{self.name}{suffix}.pth'))
        if file_path == 'official':   # TODO
            _dict = self.get_official_weights(map_location=map_location)
            last_bias_value = next(
                reversed(_dict.values()))   # TODO: make sure
            if self.num_classes != len(last_bias_value) \
                    and component != 'features':
                strict = False
                _dict.popitem()
                _dict.popitem()
        else:
            try:
                # TODO: type annotation might change?
                # dict[str, torch.Tensor]
                _dict: OrderedDict[str, torch.Tensor] = torch.load(
                    file_path, map_location=map_location, **kwargs)
            except Exception:
                print(f'{file_path=}')
                raise
        module = self._model
        match component:
            case 'features':
                module = self._model.features
                _dict = OrderedDict(
                    [(key.removeprefix('features.'), value)
                        for key, value in _dict.items()])
            case 'classifier':
                module = self._model.classifier
                _dict = OrderedDict(
                    [(key.removeprefix('classifier.'), value)
                        for key, value in _dict.items()])
            case _:
                assert component == 'full', f'{component=}'
        if inplace:
            try:
                module.load_state_dict(_dict, strict=strict)
            except RuntimeError:
                prints(f'Model {self.name} loaded from: {file_path}',
                       indent=indent)
                raise
        if verbose:
            prints(f'Model {self.name} loaded from: {file_path}',
                   indent=indent)
        if env['num_gpus']:
            self.cuda()
        return _dict

    @torch.no_grad()
    def save(self, file_path: str = None, folder_path: str = None,
             suffix: str = None, component: str = '',
             verbose: bool = False, indent: int = 0, **kwargs):
        r"""Save pretrained model weights.

        Args:
            file_path (str | None): The file path to save pretrained weights.
                Defaults to ``'{folder_path}/{self.name}{suffix}.pth'``.
            folder_path (str | None): The folder path containing model checkpoint.
                It is used when :attr:`file_path` is not provided.
                Defaults to :attr:`self.folder_path`.
            suffix (str | None): The suffix string to model weights file.
                Defaults to :attr:`self.suffix`.
            component (str): Specify which part of the weights to save.
                Choose from ``['full', 'features', 'classifier']``.
                Defaults to ``'full'``.
            verbose (bool): Whether to output auxiliary information.
                Defaults to ``False``.
            indent (int): The indent of output auxialiary information.
            **kwargs: Keyword arguments passed to :any:`torch.save`.
        """
        if file_path is None:
            folder_path = folder_path if folder_path is not None \
                else self.folder_path
            suffix = suffix if suffix is not None else self.suffix
            file_path = os.path.normpath(os.path.join(
                folder_path, f'{self.name}{suffix}.pth'))
        else:
            folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # TODO: type annotation might change? dict[str, torch.Tensor]
        module = self._model
        if component == 'features':
            module = self._model.features
        elif component == 'classifier':
            module = self._model.classifier
        else:
            assert component == '', f'{component=}'
        _dict: OrderedDict[str, torch.Tensor] = module.state_dict(
            prefix=component)
        torch.save(_dict, file_path, **kwargs)
        if verbose:
            prints(
                f'Model {self.name} saved at: {file_path}', indent=indent)

    def get_official_weights(self, url: str = None,
                             map_location: str | Callable | torch.device | dict = 'cpu',
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        r"""Get official model weights from :attr:`url`.

        Args:
            url (str | None): The link to model weights.
                Defaults to :attr:`self.model_urls[self.name]`.
            map_location (str | ~torch.torch.device | dict):
                Passed to :any:`torch.hub.load_state_dict_from_url`.
                Defaults to ``'cpu'``.
            **kwargs: Keyword arguments passed to
                :any:`torch.hub.load_state_dict_from_url`.

        Returns:
            OrderedDict[str, torch.Tensor]: The model weights OrderedDict.
        """
        url = self.model_urls[self.name] if url is None else url
        print('get official model weights from: ', url)
        return torch.hub.load_state_dict_from_url(url, map_location=map_location, **kwargs)

    # ---------------------Train and Validate--------------------- #
    # TODO: annotation and remove those arguments to be *args, **kwargs
    def _train(self, epochs: int, optimizer: Optimizer,
               lr_scheduler: _LRScheduler = None,
               lr_warmup_epochs: int = 0,
               model_ema: ExponentialMovingAverage = None,
               model_ema_steps: int = 32,
               grad_clip: float = None, pre_conditioner: None | KFAC | EKFAC = None,
               print_prefix: str = 'Train', start_epoch: int = 0, resume: int = 0,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               loader_train: torch.utils.data.DataLoader = None,
               loader_valid: torch.utils.data.DataLoader = None,
               epoch_fn: Callable[..., None] = None,
               get_data_fn: Callable[
                   ..., tuple[torch.Tensor, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               after_loss_fn: Callable[..., None] = None,
               validate_fn: Callable[..., tuple[float, float]] = None,
               save_fn: Callable[..., None] = None, file_path: str = None,
               folder_path: str = None, suffix: str = None,
               writer=None, main_tag: str = 'train', tag: str = '',
               accuracy_fn: Callable[..., list[float]] = None,
               verbose: bool = True, indent: int = 0, **kwargs):
        r"""Train the model"""
        loader_train = loader_train if loader_train is not None \
            else self.dataset.loader['train']
        get_data_fn = get_data_fn if callable(get_data_fn) else self.get_data
        loss_fn = loss_fn if callable(loss_fn) else self.loss
        validate_fn = validate_fn if callable(validate_fn) else self._validate
        save_fn = save_fn if callable(save_fn) else self.save
        accuracy_fn = accuracy_fn if callable(accuracy_fn) else self.accuracy
        kwargs['forward_fn'] = kwargs.get('forward_fn', self.__call__)
        # if not callable(iter_fn) and hasattr(self, 'iter_fn'):
        #     iter_fn = getattr(self, 'iter_fn')
        if not callable(epoch_fn) and hasattr(self, 'epoch_fn'):
            epoch_fn = getattr(self, 'epoch_fn')
        if not callable(after_loss_fn) and hasattr(self, 'after_loss_fn'):
            after_loss_fn = getattr(self, 'after_loss_fn')
        return train(module=self._model, num_classes=self.num_classes,
                     epochs=epochs, optimizer=optimizer, lr_scheduler=lr_scheduler,
                     lr_warmup_epochs=lr_warmup_epochs,
                     model_ema=model_ema, model_ema_steps=model_ema_steps,
                     grad_clip=grad_clip, pre_conditioner=pre_conditioner,
                     print_prefix=print_prefix, start_epoch=start_epoch,
                     resume=resume, validate_interval=validate_interval,
                     save=save, amp=amp,
                     loader_train=loader_train, loader_valid=loader_valid,
                     epoch_fn=epoch_fn, get_data_fn=get_data_fn,
                     loss_fn=loss_fn, after_loss_fn=after_loss_fn,
                     validate_fn=validate_fn,
                     save_fn=save_fn, file_path=file_path,
                     folder_path=folder_path, suffix=suffix,
                     writer=writer, main_tag=main_tag, tag=tag,
                     accuracy_fn=accuracy_fn,
                     verbose=verbose, indent=indent, **kwargs)

    def _validate(self, module: nn.Module = None, num_classes: int = None,
                  loader: torch.utils.data.DataLoader = None,
                  print_prefix: str = 'Validate',
                  indent: int = 0, verbose: bool = True,
                  get_data_fn: Callable[
                      ..., tuple[torch.Tensor, torch.Tensor]] = None,
                  loss_fn: Callable[..., torch.Tensor] = None,
                  writer=None, main_tag: str = 'valid',
                  tag: str = '', _epoch: int = None,
                  accuracy_fn: Callable[..., list[float]] = None,
                  **kwargs) -> tuple[float, float]:
        r"""Evaluate the model.

        Returns:
            (float, float): Accuracy and loss.
        """
        module = self._model if module is None else module
        num_classes = self.num_classes if num_classes is None else num_classes
        loader = loader or self.dataset.loader['valid']
        get_data_fn = get_data_fn or self.get_data
        loss_fn = loss_fn or self.loss
        accuracy_fn = accuracy_fn if callable(accuracy_fn) else self.accuracy
        kwargs['forward_fn'] = kwargs.get('forward_fn', self.__call__)
        return validate(module=module, num_classes=num_classes, loader=loader,
                        print_prefix=print_prefix,
                        indent=indent, verbose=verbose,
                        get_data_fn=get_data_fn,
                        loss_fn=loss_fn,
                        writer=writer, main_tag=main_tag, tag=tag,
                        _epoch=_epoch, accuracy_fn=accuracy_fn, **kwargs)

    def _compare(self, peer: nn.Module = None,
                 loader: torch.utils.data.DataLoader = None,
                 print_prefix: str = 'Validate',
                 indent: int = 0, verbose: bool = True,
                 get_data_fn: Callable[
                     ..., tuple[torch.Tensor, torch.Tensor]] = None,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 **kwargs) -> tuple[float, float]:
        loader = loader or self.dataset.loader['valid']
        get_data_fn = get_data_fn or self.get_data
        criterion = criterion or self.criterion
        return compare(self, peer, loader,
                       print_prefix, indent, verbose,
                       get_data_fn, criterion=self.criterion, **kwargs)

    # ----------------------------Utility--------------------------- #

    def get_data(self, data, **kwargs):
        r"""Process data. Defaults to be :attr:`self.dataset.get_data`.
        If :attr:`self.dataset` is ``None``, return :attr:`data` directly.

        Args:
            data (Any): Unprocessed data.
            **kwargs: Keyword arguments passed to
                :attr:`self.dataset.get_data()`.

        Returns:
            Any: Processed data.
        """
        if self.dataset is not None:
            return self.dataset.get_data(data, **kwargs)
        else:
            return data

    def accuracy(self, _output: torch.Tensor, _label: torch.Tensor,
                 num_classes: int = None,
                 topk: tuple[int] = (1, 5)) -> list[float]:
        r"""Computes the accuracy over the k top predictions
        for the specified values of k.

        Args:
            _output (torch.Tensor): The batched logit tensor with shape ``(N, C)``.
            _label (torch.Tensor): The batched label tensor with shape ``(N)``.
            num_classes (int): Number of classes. Defaults to :attr:`self.num_classes`.
            topk (~collections.abc.Iterable[int]): Which top-k accuracies to show.
                Defaults to ``(1, 5)``.

        Returns:
            list[float]: Top-k accuracies.

        Note:
            The implementation is in :func:`trojanzoo.utils.model.accuracy`.
        """
        num_classes = num_classes or self.num_classes
        return accuracy(_output, _label, num_classes, topk)

    def activate_params(self, params: Iterator[nn.Parameter] = []) -> None:
        r"""Set ``requires_grad=True`` for selected :attr:`params` of :attr:`module`.
        All other params are frozen.

        Args:
            params (~collections.abc.Iterator[torch.nn.parameter.Parameter]):
                The parameters to ``requires_grad``.
                Defaults to ``[]``.
        """
        return activate_params(self._model, params)

    # Need to overload for other packages (GNN)
    # since they are calling their own nn.DataParallel.
    # TODO: nn.parallel.DistributedDataParallel
    @staticmethod
    def get_parallel_model(_model: _Model) -> _Model | nn.DataParallel:
        r"""Get the parallel model if there are more than 1 GPU avaiable.

        Warning:
            :any:`torch.nn.DataParallel` would be deprecated according to
            https://github.com/pytorch/pytorch/issues/65936.
            We need to consider using
            :any:`torch.nn.parallel.DistributedDataParallel` instead.

        Args:
            _model (_Model): The non-parallel model.
        Returns:
            _Model | nn.DataParallel: The parallel model if there are more than 1 GPU avaiable.
        """
        if env['num_gpus'] > 1:
            return nn.DataParallel(_model)
        return _model

    def summary(self, depth: int = None, verbose: bool = True,
                indent: int = 0, **kwargs):
        r"""Prints a string summary of the model instance by calling
        :func:`trojanzoo.utils.module.BasicObject.summary()`
        and :func:`trojanzoo.utils.model.summary()`.

        Args:
            depth (int): Passed to :func:`trojanzoo.utils.model.summary()`.
                If ``None``, set as ``env['verbose']``.
                If still ``None``, set as ``1``.
                Defaults to ``None``.
            verbose (bool): Passed to :func:`trojanzoo.utils.model.summary()`.
                Defaults to ``True``.
            indent (int): Passed to :func:`trojanzoo.utils.module.BasicObject.summary()`
                and passed to :func:`trojanzoo.utils.model.summary()` with ``10`` more.
                Defaults to ``0``.
            **kwargs: Passed to :func:`trojanzoo.utils.model.summary()`.
        """
        super().summary(indent=indent)
        if depth is None:
            depth = env['verbose']
        if depth is None:
            depth = 1
        summary(self._model, depth=depth, verbose=verbose,
                indent=indent + 10, **kwargs)
        prints('-' * 20, indent=indent + 10)

    # -------------------------------Reload---------------------------- #

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        See Also:
            :any:`torch.nn.Module.train`.
        """
        self._model.train(mode=mode)
        self.model.train(mode=mode)
        return self

    def eval(self):
        r"""Sets the module in evaluation mode.

        See Also:
            :any:`torch.nn.Module.eval`.
        """
        self._model.eval()
        self.model.eval()
        return self

    def cpu(self):
        r"""Moves all model parameters and buffers to the CPU.

        See Also:
            :any:`torch.nn.Module.cpu`.
        """
        self._model.cpu()
        self.model.cpu()
        return self

    def cuda(self, device: None | int | torch.device = None):
        r"""Moves all model parameters and buffers to the GPU.

        See Also:
            :any:`torch.nn.Module.cuda`.
        """
        self._model.cuda(device=device)
        self.model.cuda(device=device)
        return self

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets gradients of all model parameters to zero.

        See Also:
            :any:`torch.nn.Module.zero_grad`.
        """
        return self._model.zero_grad(set_to_none=set_to_none)

    def state_dict(self, destination: Mapping[str, torch.Tensor] = None,
                   prefix: str = '', keep_vars: bool = False):
        r"""Returns a dictionary containing a whole state of the module.

        See Also:
            :any:`torch.nn.Module.state_dict`.
        """
        return self._model.state_dict(destination=destination, prefix=prefix,
                                      keep_vars=keep_vars)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor],
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict`
        into this module and its descendants.

        See Also:
            :any:`torch.nn.Module.load_state_dict`.
        """
        return self._model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse: bool = True):
        r"""Returns an iterator over module parameters.

        See Also:
            :any:`torch.nn.Module.parameters`.
        """
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        See Also:
            :any:`torch.nn.Module.named_parameters`.
        """
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def children(self):
        r"""Returns an iterator over immediate children modules.

        See Also:
            :any:`torch.nn.Module.children`.
        """
        return self._model.children()

    def named_children(self):
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        See Also:
            :any:`torch.nn.Module.named_children`.
        """
        return self._model.named_children()

    def modules(self):
        r"""Returns an iterator over all modules in the network.

        See Also:
            :any:`torch.nn.Module.modules`.
        """
        return self._model.modules()

    def named_modules(self, memo: None | set[nn.Module] = None,
                      prefix: str = ''
                      ) -> Generator[tuple[str, nn.Module], None, None]:
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        See Also:
            :any:`torch.nn.Module.named_modules`.
        """
        return self._model.named_modules(memo=memo, prefix=prefix)

    def apply(self, fn: Callable[['nn.Module'], None]):
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model

        See Also:
            :any:`torch.nn.Module.apply`.
        """
        return self._model.apply(fn)

    def requires_grad_(self, requires_grad: bool = True):
        r"""Change if autograd should record operations on parameters in this
        module.

        See Also:
            :any:`torch.nn.Module.requires_grad_`.
        """
        return self._model.requires_grad_(requires_grad=requires_grad)

    # ----------------------------------------------------------------- #

    @torch.no_grad()
    def remove_misclassify(self, data: tuple[torch.Tensor, torch.Tensor],
                           **kwargs):
        r"""Remove misclassified samples in a data batch.

        Args:
            data (tuple[torch.Tensor, torch.Tensor]):
                The input and label to process with shape ``(N, *)`` and ``(N)``.
            **kwargs: Keyword arguments passed to :meth:`get_data`.

        Returns:
            (torch.Tensor, torch.Tensor):
                The processed input and label with shape ``(N - k, *)`` and ``(N - k)``.
        """
        _input, _label = self.get_data(data, **kwargs)
        _classification = self.get_class(_input)
        repeat_idx = _classification.eq(_label)
        return _input[repeat_idx], _label[repeat_idx]

    def generate_target(self, _input: torch.Tensor, idx: int = 1,
                        same: bool = False) -> torch.Tensor:
        r"""Generate target labels of a batched input based on
            the classification confidence ranking index.

        Args:
            _input (torch.Tensor): The input tensor.
            idx (int): The classification confidence
                rank of target class.
                Defaults to ``1``.
            same (bool): Generate the same label
                for all samples using mod.
                Defaults to ``False``.

        Returns:
            torch.Tensor:
                The generated target label with shape ``(N)``.

        See Also:
            The implementation is in
            :func:`trojanzoo.utils.model.generate_target()`.
        """
        return generate_target(self, _input, idx, same)

    # --------------------- Undocumented Methods ------------------------- #

    def get_parameter_from_name(self, name: str = 'full'
                                ) -> Iterator[nn.Parameter]:
        match name:
            case 'features':
                params = self._model.features.parameters()
            case 'classifier' | 'partial':
                params = self._model.classifier.parameters()
            case 'full':
                params = self._model.parameters()
            case _:
                raise NotImplementedError(f'{name=}')
        return params

    def __call__(self, _input: torch.Tensor, amp: bool = False,
                 **kwargs) -> torch.Tensor:
        if amp:
            with torch.cuda.amp.autocast():
                return self.get_logits(_input, **kwargs)
        return self.get_logits(_input, **kwargs)


def add_argument(parser: argparse.ArgumentParser,
                 model_name: None | str = None,
                 model: None | str | Model = None,
                 config: Config = config,
                 class_dict: dict[str, type[Model]] = {}
                 ) -> argparse._ArgumentGroup:
    r"""
    | Add model arguments to argument parser.
    | For specific arguments implementation, see :meth:`Model.add_argument()`.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments.
        model_name (str): The model name.
        model (str | Model): The model instance or model name
            (as the alias of `model_name`).
        config (Config): The default parameter config,
            which contains the default dataset and model name if not provided.
        class_dict (dict[str, type[Model]]):
            Map from model name to model class.
            Defaults to ``{}``.

    Returns:
        argparse._ArgumentGroup: The argument group.
    """
    dataset_name = get_name(arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.full_config['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model,
                          arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)[
            'model']['default_model']
    model_name = get_model_class(model_name, class_dict=class_dict)

    group = parser.add_argument_group(
        '{yellow}model{reset}'.format(**ansi), description=model_name)
    model_class_name = get_model_class(model_name, class_dict=class_dict)
    try:
        ModelType = class_dict[model_class_name]
    except KeyError:
        print(f'{model_class_name} not in \n{list(class_dict.keys())}')
        raise
    return ModelType.add_argument(group)


def create(model_name: None | str = None, model: None | str | Model = None,
           dataset_name: None | str = None, dataset: None | str | Dataset = None,
           config: Config = config,
           class_dict: dict[str, type[Model]] = {},
           **kwargs) -> Model:
    r"""
    | Create a model instance.
    | For arguments not included in :attr:`kwargs`,
      use the default values in :attr:`config`.
    | The default value of :attr:`folder_path` is
      ``'{model_dir}/{dataset.data_type}/{dataset.name}'``.
    | For model implementation, see :class:`Model`.

    Args:
        model_name (str): The model name.
        model (str | Model): The model instance or model name
            (as the alias of `model_name`).
        dataset_name (str): The dataset name.
        dataset (str | trojanzoo.datasets.Dataset):
            Dataset instance or dataset name
            (as the alias of `dataset_name`).
        config (Config): The default parameter config.
        class_dict (dict[str, type[Model]]):
            Map from model name to model class.
            Defaults to ``{}``.
        **kwargs: The keyword arguments
            passed to model init method.

    Returns:
        Model: The model instance.
    """
    dataset_name = get_name(
        name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model,
                          arg_list=['-m', '--model'])
    if dataset_name is None:
        dataset_name = config.full_config['dataset']['default_dataset']
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)[
            'model']['default_model']
    result = config.get_config(dataset_name=dataset_name)[
        'model'].update(kwargs)
    model_name = model_name if model_name is not None \
        else result['default_model']

    name_list = [name for sub_list in get_available_models(
        class_dict=class_dict).values()
        for name in sub_list]
    name_list = sorted(name_list)
    assert model_name in name_list, f'{model_name} not in \n{name_list}'
    model_class_name = get_model_class(model_name, class_dict=class_dict)
    try:
        ModelType = class_dict[model_class_name]
    except KeyError:
        print(f'{model_class_name} not in \n{list(class_dict.keys())}')
        raise

    if 'folder_path' not in result.keys() and isinstance(dataset, Dataset):
        result['folder_path'] = os.path.join(result['model_dir'],
                                             dataset.data_type,
                                             dataset.name)
    return ModelType(name=model_name, dataset=dataset, dataset_name=dataset_name, **result)


def output_available_models(class_dict: dict[str, type[Model]] = {},
                            indent: int = 0) -> None:
    r"""Output all available model names.

    Args:
        class_dict (dict[str, type[Model]]): Map from model name to model class.
            Defaults to ``{}``.
        indent (int): The space indent for the entire string.
            Defaults to ``0``.
    """
    names_dict = get_available_models(class_dict)
    for k in sorted(names_dict.keys()):
        prints('{yellow}{k}{reset}'.format(k=k, **ansi), indent=indent)
        prints(names_dict[k], indent=indent + 10)
        print()


def get_available_models(class_dict: dict[str, type[Model]] = {}
                         ) -> dict[str, list[str]]:
    return {k: v.available_models for k, v in class_dict.items()}


def get_model_class(name: str, class_dict: dict[str, type[Model]] = {}) -> str:
    correct_name: str = None
    for class_name in class_dict.keys():
        if class_name in name.lower() and \
                (correct_name is None or
                 len(class_name) > len(correct_name)):
            correct_name = class_name
    if correct_name is not None:
        return correct_name
    raise KeyError(f'{name} not in {list(class_dict.keys())}')
