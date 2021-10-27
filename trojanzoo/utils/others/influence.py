#!/usr/bin/env python3

from trojanzoo.environ import env
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import TYPE_CHECKING
from trojanzoo.models import Model    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.utils.data


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

    def up_loss(self, z: torch.Tensor = None, z_label: torch.Tensor = None, v: torch.Tensor = None,
                z_test: torch.Tensor = None, z_test_label: torch.Tensor = None, v_test: torch.Tensor = None,
                hess_inv: torch.Tensor = None) -> list[float]:
        if v is None:
            v = self.calc_v(z, z_label)
        if (z_test is None and z_test_label is None):
            z_test = z
            z_test_label = z_label
        if v_test is None:
            if (z is None and z_label is None) or (z is z_test and z_label is z_test_label):
                v_test = v
            else:
                v_test = self.calc_v(z_test, z_test_label)
        s_test = v_test @ hess_inv
        return (v * s_test).sum(dim=1).detach().cpu().tolist()

    def calc_v(self, z: torch.Tensor, z_test_label: torch.Tensor) -> torch.Tensor:
        def func(weight: nn.Parameter) -> torch.Tensor:
            del self.module.weight
            self.module.weight = weight
            _output = self.model(_input=z)
            return self.criterion_vec(_output, z_test_label)
        jocobian: torch.Tensor = torch.autograd.functional.jacobian(func, self.parameter)
        return jocobian.flatten(1)

    # def calc_s_test(self, v_test: torch.Tensor, hess_inv: torch.Tensor = None) -> torch.Tensor:
    #     if isinstance(hess_inv, torch.Tensor):
    #         return v_test @ hess_inv

    def calc_H(self, loader: torch.utils.data.DataLoader = None, eps=1e-5, **kwargs) -> torch.Tensor:
        loader = loader if loader is not None else self.dataset.loader['train']
        hess_list: list[torch.Tensor] = []
        batch_num = max(self.sample_size // loader.batch_size, 1)
        for i, data in enumerate(loader):
            if i >= batch_num:
                break
            _input, _label = self.model.get_data(data, **kwargs)
            _feats = self.model.get_final_fm(_input)    # TODO
            hess = self.compute_hess(_feats, _label).detach().cpu()  # (N, D, D)
            hess_list.append(hess)
        hess = torch.stack(hess_list).mean(dim=0).to(env['device'])  # (D, D)
        return (hess + hess.t()) / 2 + eps * torch.eye(len(hess), device=hess.device)   # numerical robust

    def compute_fim(self, _feats: torch.Tensor) -> torch.Tensor:
        # _feats: (N, D)
        def func(weight: nn.Parameter) -> torch.Tensor:
            del self.module.weight  # (D, class_num)
            self.module.weight = weight
            log_prob = F.log_softmax(self.module(_feats))  # (N, class_num)
            return log_prob.mean(dim=0)  # (class_num)
        jacobian: torch.Tensor = torch.autograd.functional.jacobian(func, self.parameter)
        jacobian = jacobian.flatten(start_dim=2)  # (class_num, D*class_num)
        prob = F.softmax(self.module(_feats)).unsqueeze(-1).unsqueeze(-1)  # (N, class_num, 1, 1)
        hess = prob * jacobian.unsqueeze(-1) * jacobian.unsqueeze(-2)  # (N, class_num, D*class_num, D*class_num)
        hess = hess.sum(dim=1)  # (N, D*class_num, D*class_num)
        return hess

    def compute_hess(self, _feats: torch.Tensor, _label: torch.Tensor) -> torch.Tensor:
        # _feats: (N, D)
        def func(weight: nn.Parameter) -> torch.Tensor:
            del self.module.weight
            self.module.weight = weight
            return self.model.criterion(self.module(_feats), _label)
        hess: torch.Tensor = torch.autograd.functional.hessian(func, self.parameter)
        return hess.flatten(-2, -1).flatten(0, 1)
        # def func(weight):
        #     del self.module.weight
        #     self.module.weight = weight
        #     return self.model.criterion(self.module(_feats), _label)
        # return torch.autograd.functional.hessian(func, self.parameter).flatten(-2, -1).flatten(0, 1)

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
