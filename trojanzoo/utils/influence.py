#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanzoo.models import Model
from .data import sample_batch
from .tensor import to_tensor, to_numpy
import torch
import torch.autograd
import torch.nn as nn
import torch.utils.data
import numpy as np
# from scipy.optimize import fmin_ncg
from scipy.optimize import minimize
from typing import Callable


class InfluenceFunction():
    def __init__(self, model: Model, sample_size: int = 64, **kwargs):
        self.sample_size = sample_size
        self.model = model
        self.dataset = self.model.dataset
        self.module = self.get_module()
        self.parameter = self.get_parameter()
        self.criterion_vec = nn.CrossEntropyLoss(weight=self.dataset.loss_weights, reduction='none')

    def get_module(self) -> nn.Linear:
        return self.model._model.classifier[-1]

    def get_parameter(self) -> nn.Parameter:
        return self.module.weight

    def up_loss(self, z: torch.Tensor, z_label: torch.Tensor,
                z_test: torch.Tensor = None, z_test_label: torch.Tensor = None,
                hess_inv: torch.Tensor = None) -> list[float]:
        v = self.calc_v(z, z_label)
        v_test = v
        if z_test is None and z_test_label is None:
            z_test = z
            z_test_label = z_label
        if not (z is z_test and z_label is z_test_label):
            v_test = self.calc_v(z_test, z_test_label)
        s_test = self.calc_s_test(v_test=v_test, hess_inv=hess_inv)
        return (v * s_test).detach().cpu().tolist()

    def calc_v(self, z_test: torch.Tensor, z_test_label: torch.Tensor) -> torch.Tensor:
        def func(weight):
            del self.module.weight
            self.module.weight = weight
            _output = self.model(_input=z_test)
            return self.criterion_vec(_output, z_test_label)
        return torch.autograd.functional.jacobian(func, self.parameter).flatten(1)

    def calc_s_test(self, v_test: torch.Tensor, hess_inv: torch.Tensor = None) -> torch.Tensor:
        if isinstance(hess_inv, torch.Tensor):
            return v_test @ hess_inv

    def calc_H(self, loader: torch.utils.data.DataLoader = None, eps=1e-5) -> torch.Tensor:
        loader = loader if loader is not None else self.dataset.loader['train']
        hess_list: list[torch.Tensor] = []
        batch_num = max(self.sample_size // loader.batch_size, 1)
        for i, data in enumerate(loader):
            if i >= batch_num:
                break
            _input, _label = self.model.get_data(data)
            _feats = self.model.get_final_fm(_input)    # TODO
            hess = self.compute_hessian(_feats, _label)
            hess_list.append(hess)
        hess = torch.cat(hess_list).mean(dim=0)
        return (hess + hess.t()) / 2 + eps * torch.eye(len(hess), device=hess.device)   # numerical robust

    def compute_hessian(self, _feats: torch.Tensor, _label: torch.Tensor) -> torch.Tensor:
        def func(weight):
            del self.module.weight
            self.module.weight = weight
            return self.model.criterion(self.module(_feats), _label)
        return torch.autograd.functional.hessian(func, self.parameter).flatten(-2, -1).flatten(0, 1)

    # def calc_Ht(self, t: torch.Tensor, _input: torch.Tensor = None, _label: torch.Tensor = None) -> torch.Tensor:
    #     if _input is None and _label is None:
    #         dataset = self.dataset.loader['train'].dataset
    #         data = sample_batch(dataset, self.sample_size)
    #         _input, _label = self.model.get_data(data)
    #     loss = self.model.loss(_input=_input, _label=_label)

    #     self.parameter.requires_grad()
    #     grad = torch.autograd.grad(loss, self.parameter,
    #                                create_graph=True)[0].flatten()    # (D)
    #     Ht = torch.autograd.grad(grad @ t, self.parameter)[0].flatten().detach()
    #     self.parameter.requires_grad(False)
    #     return Ht    # (D)
