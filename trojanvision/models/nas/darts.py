#!/usr/bin/env python3

"""
CUDA_VISIBLE_DEVICES=0 python examples/train.py --color --verbose 1 --dataset cifar10 --model darts --supernet --arch_search --arch_unrolled --layers 8 --init_channels 16 --batch_size 64 --lr 0.025 --lr_scheduler --lr_min 1e-3 --grad_clip 5.0 --epochs 50
"""  # noqa: E501

import trojanvision.utils.model_archs.darts as darts
from trojanvision.datasets import ImageSet
from trojanvision.models.imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs.darts import FeatureExtractor, AuxiliaryHead, Genotype, genotypes
from trojanvision.utils.model_archs.darts.operations import PRIMITIVES

import torch
import torch.nn as nn
from torchvision.datasets.utils import download_file_from_google_drive
import os
import itertools
from collections import OrderedDict

from typing import TYPE_CHECKING
from trojanzoo.utils.fim import KFAC, EKFAC
from trojanzoo.utils.model import ExponentialMovingAverage
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data
import argparse  # TODO: python 3.10
from collections.abc import Callable
if TYPE_CHECKING:
    pass

url = {
    'cifar10': '1Y13i4zKGKgjtWBdC0HWLavjO7wvEiGOc',
    'ptb': '1Mt_o6fZOlG-VDF3Q5ModgnAJ9W6f_av2',
    'imagenet': '1AKr6Y_PoYj7j0Upggyzc26W0RVdg4CVX'
}


def _concat(xs: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.flatten() for x in xs])


class _DARTS(_ImageModel):
    def __init__(self, auxiliary: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.features: FeatureExtractor | darts.search.FeatureExtractor
        self.auxiliary_head: nn.Sequential = None
        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C=self.features.aux_dim, num_classes=self.num_classes)
        num_features = kwargs.get('num_features', [self.features.feats_dim])
        self.classifier = self.define_classifier(num_features=num_features, num_classes=self.num_classes)

    @staticmethod
    def define_features(supernet: bool = False,
                        genotype: Genotype = genotypes.darts,
                        init_channels: int = 36, layers: int = 20,
                        dropout_p: float = 0.2, **kwargs
                        ) -> FeatureExtractor | darts.search.FeatureExtractor:
        if supernet:
            return darts.search.FeatureExtractor(init_channels, layers, **kwargs)
        return FeatureExtractor(genotype, init_channels, layers, dropout_p, **kwargs)


class DARTS(ImageModel):
    r"""DARTS-like models used in Neural Architecture Search.

    :Available model names:

        .. code-block:: python3

            ['darts']

    See Also:
        * paper: `DARTS\: Differentiable Architecture Search`_
        * code: https://github.com/quark0/darts

    Args:
        supernet (bool): Whether to use supernet (mixed operations).
            Defaults to ``False``.
        model_arch (str): Genotype name in ``trojanvision.utils.model_archs.genotypes`` to use.
            Defaults to ``'darts'``.

            * ``'amoebanet', 'amoebanet_adapt'``
            * ``'darts_v1', 'darts_v2'('darts')``
            * ``'drnas_cifar10'('drnas'), 'drnas_imagenet'``
            * ``'enas', 'enas_adapt'``
            * ``'nasnet', 'nasnet_adapt'``
            * ``'pc_darts_cifar'('pc_darts'), 'pc_darts_image'``
            * ``'pdarts'``
            * ``'robust_darts'``
            * ``'sgas'``
            * ``'snas_mild', 'snas_adapt'``
            * ``'random'``
            * ``'diy_deep', 'diy_noskip', 'diy_deep_noskip'``
        layers (int): Total number of layers. Defaults to ``20``.
        init_channels (int): :attr:`out_channel` of stem conv layer.
            Defaults to ``36``.
        dropout_p (float): Dropout probability.
            Defaults to ``0.2``.
        auxiliary (bool): Whether to use auxiliary classifier.
            Defaults to ``False``.
        auxiliary_weight (float): Loss weight of auxiliary classifier.
            Defaults to ``0.4``.
        arch_search (bool): Whether to search supernet architecture weight parameters.
            Defaults to ``False``.
        use_full_train_set (bool): Whether to use full training data during architecture search.
            Defaults to ``False``.
        arch_lr (float): Learning rate for architecture optimizer.
            Defaults to ``3e-4``
        arch_weight_decay (float): Weight decay for architecture optimizer.
            Defaults to ``1e-3``.
        arch_unrolled (bool): Whether to use one-step unrolled validation loss (darts-v2).
            Defaults to ``False``.

    Attributes:
        genotype (Genotype): Genotype of cell architecture.

    Note:
        The implementation of DARTS model is in ``trojanvision.utils.model_archs.darts``

    .. _DARTS\: Differentiable Architecture Search:
        https://arxiv.org/abs/1806.09055
    """
    available_models = ['darts']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--supernet', action='store_true', help='whether to use supernet')
        group.add_argument('--model_arch', help='genotype name (default: "darts")')
        group.add_argument('--layers', type=int, help='total number of layers (default: 20)')
        group.add_argument('--init_channels', type=int, help='out_channel of stem conv layer (default: 36)')
        group.add_argument('--dropout_p', type=float, help='dropout probability (default: 0.2)')
        group.add_argument('--auxiliary', action='store_true', help='whether to use auxiliary classifier')
        group.add_argument('--auxiliary_weight', type=float,
                           help='loss weight of auxiliary classifier (default: 0.4)')
        group.add_argument('--arch_search', action='store_true',
                           help='whether to search supernet architecture weight parameters')
        group.add_argument('--use_full_train_set', action='store_true',
                           help='whether to use full training data during architecture search')
        group.add_argument('--arch_lr', type=float,
                           help='learning rate for architecture optimizer (default: 3e-4)')
        group.add_argument('--arch_weight_decay', type=float,
                           help='weight decay for architecture optimizer (default: 1e-3)')
        group.add_argument('--arch_unrolled', action='store_true', default=False,
                           help='whether to use one-step unrolled validation loss (darts-v2)')
        return group

    def __init__(self, name: str = 'darts', model_arch: str = 'darts',
                 layers: int = 20, init_channels: int = 36, dropout_p: float = 0.2,
                 auxiliary: bool = False, auxiliary_weight: float = 0.4,
                 genotype: Genotype = None, model: type[_DARTS] = _DARTS,
                 supernet: bool = False, arch_search: bool = False,
                 use_full_train_set: bool = False,
                 arch_lr: float = 3e-4, arch_weight_decay=1e-3,
                 arch_unrolled: bool = False,
                 primitives=PRIMITIVES, **kwargs):
        # TODO: ImageNet parameter settings
        self.supernet = supernet
        self.arch_search = arch_search
        if supernet:
            name += '_supernet'
        if not supernet and genotype is None:
            model_arch = model_arch.lower()
            name = model_arch
            try:
                genotype = getattr(genotypes, model_arch)
            except AttributeError:
                print('Available Model Architectures: ')
                model_arch_list = [element for element in dir(genotypes)
                                   if '__' not in element and
                                   element not in ['Genotype', 'namedtuple']]
                print(model_arch_list)
                raise
        self.layers = layers
        self.init_channels = init_channels
        self.dropout_p = dropout_p
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.arch_lr = arch_lr
        self.arch_weight_decay = arch_weight_decay
        self.arch_unrolled = arch_unrolled
        super().__init__(name=name, layers=layers, init_channels=init_channels, dropout_p=dropout_p,
                         genotype=genotype, model=model,
                         auxiliary=auxiliary, primitives=primitives,
                         supernet=supernet, **kwargs)
        self._model: _DARTS
        self.param_list['darts'] = ['supernet', 'layers', 'init_channels']
        if not supernet:
            self.param_list['darts'].extend(['dropout_p', 'genotype'])
        if auxiliary:
            self.param_list['darts'].insert(1, 'auxiliary_weight')

        if supernet and arch_search:
            self.use_full_train_set = use_full_train_set
            for alpha in self.arch_parameters():
                alpha.requires_grad_()
            train2, train3 = self.dataset.split_dataset(
                self.dataset.get_dataset('train'),
                percent=0.5)
            self.train2 = self.dataset.get_dataloader(
                mode='train', dataset=train2)
            self.train3 = self.dataset.get_dataloader(
                mode='train', dataset=train3)
            self.valid_iterator = itertools.cycle(self.train3)
            self.arch_optimizer = torch.optim.Adam(self.arch_parameters(),
                                                   lr=arch_lr, betas=(0.5, 0.999),
                                                   weight_decay=arch_weight_decay)
            self.param_list['arch_search'] = ['use_full_train_set', 'arch_optimizer']

    @property
    def genotype(self) -> Genotype:
        return self._model.features.genotype() if self.supernet else self._model.features.genotype

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, amp: bool = False, **kwargs) -> torch.Tensor:
        if self.auxiliary:
            assert isinstance(self._model.auxiliary_head, nn.Sequential)
            if amp:
                with torch.cuda.amp.autocast():
                    return self.loss_with_aux(_input, _label)
            return self.loss_with_aux(_input, _label)
        return super().loss(_input, _label, _output, amp=amp, **kwargs)

    def loss_with_aux(self, _input: torch.Tensor = None, _label: torch.Tensor = None) -> torch.Tensor:
        feats, feats_aux = self._model.features.forward_with_aux(self._model.preprocess(_input))
        logits: torch.Tensor = self._model.classifier(self._model.flatten(self._model.pool(feats)))
        logits_aux: torch.Tensor = self._model.auxiliary_head(feats_aux)
        return super().loss(_output=logits, _label=_label) \
            + self.auxiliary_weight * super().loss(_output=logits_aux, _label=_label)

    def load(self, *args, strict: bool = False, **kwargs):
        return super().load(*args, strict=strict, **kwargs)

    def get_official_weights(self, dataset: str = None, **kwargs) -> OrderedDict[str, torch.Tensor]:
        assert str(self.genotype) == str(genotypes.darts)
        if dataset is None and isinstance(self.dataset, ImageSet):
            dataset = self.dataset.name
        file_name = f'darts_{dataset}.pt'
        folder_path = os.path.join(torch.hub.get_dir(), 'darts')
        file_path = os.path.join(folder_path, file_name)
        download_file_from_google_drive(file_id=url[dataset], root=folder_path, filename=file_name)
        print('get official model weights from Google Drive: ', url[dataset])
        _dict: OrderedDict[str, torch.Tensor] = torch.load(file_path,
                                                           map_location='cpu')
        if 'state_dict' in _dict.keys():
            _dict = _dict['state_dict']

        new_dict: OrderedDict[str, torch.Tensor] = self.state_dict()
        old_keys = list(_dict.keys())
        new_keys = list(new_dict.keys())
        new2old: dict[str, str] = {}
        i = 0
        j = 0
        while(i < len(new_keys) and j < len(old_keys)):
            if 'num_batches_tracked' in new_keys[i]:
                i += 1
                continue
            if 'auxiliary_head' not in new_keys[i] and 'auxiliary_head' in old_keys[j]:
                j += 1
                continue
            new2old[new_keys[i]] = old_keys[j]
            i += 1
            j += 1
        for i, key in enumerate(new_keys):
            if 'num_batches_tracked' in key:
                new_dict[key] = torch.tensor(0)
            else:
                new_dict[key] = _dict[new2old[key]]
        return new_dict

    def arch_parameters(self) -> list[torch.Tensor]:
        return self._model.features.arch_parameters()

    def named_arch_parameters(self) -> list[tuple[str, torch.Tensor]]:
        return self._model.features.named_arch_parameters()

    def _train(self, epochs: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
               adv_train: bool = None,
               lr_warmup_epochs: int = 0,
               model_ema: ExponentialMovingAverage = None,
               model_ema_steps: int = 32,
               grad_clip: float = None, pre_conditioner: None | KFAC | EKFAC = None,
               print_prefix: str = 'Epoch', start_epoch: int = 0, resume: int = 0,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               loader_train: torch.utils.data.DataLoader = None,
               loader_valid: torch.utils.data.DataLoader = None,
               epoch_fn: Callable[..., None] = None,
               get_data_fn: Callable[...,
                                     tuple[torch.Tensor, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               after_loss_fn: Callable[..., None] = None,
               validate_fn: Callable[..., tuple[float, float]] = None,
               save_fn: Callable[..., None] = None, file_path: str = None,
               folder_path: str = None, suffix: str = None,
               writer=None, main_tag: str = 'train', tag: str = '',
               accuracy_fn: Callable[..., list[float]] = None,
               verbose: bool = True, indent: int = 0, **kwargs):
        get_data_fn = get_data_fn or self.get_data
        validate_fn = validate_fn or self._validate
        if self.arch_search:
            if not self.use_full_train_set:
                loader_train = loader_train or self.train2
            if self.arch_unrolled:
                self.optimizer = optimizer
                # self.lr_scheduler = lr_scheduler

            get_data_old = get_data_fn
            validate_old = validate_fn

            def get_data(data: tuple[torch.Tensor, torch.Tensor], adv_train: bool = False,
                         mode: str = 'train', **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
                _input, _label = get_data_old(data, adv_train=adv_train, **kwargs)
                if mode == 'train':
                    data_valid = next(self.valid_iterator)
                    input_valid, label_valid = get_data_old(data_valid, adv_train=adv_train, **kwargs)
                    self.arch_optimizer.zero_grad()
                    if self.arch_unrolled:
                        self._backward_step_unrolled(_input, _label, input_valid, label_valid)
                    else:
                        loss = self.loss(input_valid, label_valid)
                        loss.backward(inputs=self.arch_parameters())
                    self.arch_optimizer.step()
                return _input, _label

            def _validate(adv_train: bool = None,
                          loader: torch.utils.data.DataLoader = None,
                          **kwargs) -> tuple[float, float]:
                print(self.genotype)
                # if not self.use_full_train_set:
                #     super()._validate(loader=self.train3,
                #                       adv_train=adv_train,
                #                       print_prefix='TrainVal', **kwargs)
                return validate_old(loader=loader, adv_train=adv_train, **kwargs)

            get_data_fn = get_data
            validate_fn = _validate

        return super()._train(epochs=epochs, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              adv_train=adv_train,
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

    def _backward_step_unrolled(self, input_train: torch.Tensor, target_train: torch.Tensor,
                                input_valid: torch.Tensor, target_valid: torch.Tensor,
                                optimizer: Optimizer = None):
        optimizer = optimizer or self.optimizer
        eta = optimizer.param_groups[0]['lr']
        # if self.lr_scheduler is not None:
        #     eta = self.lr_scheduler.get_last_lr()
        w = self.state_dict()
        self._compute_unrolled_model(input_train, target_train, eta,
                                     optimizer=optimizer)
        unrolled_loss = self.loss(input_valid, target_valid)
        unrolled_loss.backward()
        dalpha = [v.grad for v in self.arch_parameters()]
        vector = [v.grad.data for v in self.parameters()]

        self.load_state_dict(w)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig, alpha=eta)

        for v, g in zip(self.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.copy_(g)

    # def _construct_model_from_theta(self, theta):
    #     model_new = self.model.new()
    #     model_dict = self.model.state_dict()

    #     params, offset = {}, 0
    #     for k, v in self.model.named_parameters():
    #         v_length = np.prod(v.size())
    #         params[k] = theta[offset: offset + v_length].view(v.size())
    #         offset += v_length

    #     assert offset == len(theta)
    #     model_dict.update(params)
    #     model_new.load_state_dict(model_dict)
    #     return model_new.cuda()

    def _compute_unrolled_model(self, input: torch.Tensor, target: torch.Tensor,
                                eta: float, optimizer: Optimizer = None):
        optimizer = optimizer or self.optimizer
        weight_decay = optimizer.param_groups[0]['weight_decay']
        loss = self.loss(input, target)
        gtheta = torch.autograd.grad(loss, self.parameters())
        with torch.no_grad():
            dtheta = [grad + weight_decay * param for grad, param in zip(gtheta, self.parameters())]
            try:
                momentum = optimizer.param_groups[0]['momentum']
                dtheta = [dtheta_i + momentum * optimizer.state[v]['momentum_buffer']
                          for dtheta_i, v in zip(dtheta, self.model.parameters())]
            except KeyError:
                pass
            for param, delta in zip(self.parameters(), dtheta):
                param.data.sub_(delta, alpha=eta)

    def _hessian_vector_product(self, vector: list[torch.Tensor],
                                _input: torch.Tensor, target: torch.Tensor,
                                r: float = 1e-2
                                ) -> list[torch.Tensor]:
        arch_params = self.arch_parameters()
        R = r / _concat(vector).norm()
        for p, v in zip(self.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.loss(_input, target)
        grads_p = list(torch.autograd.grad(loss, arch_params, allow_unused=True))

        for p, v in zip(self.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = self.loss(_input, target)
        grads_n = list(torch.autograd.grad(loss, arch_params, allow_unused=True))

        for p, v in zip(self.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y) / (2 * R) for x, y in zip(grads_p, grads_n)]
