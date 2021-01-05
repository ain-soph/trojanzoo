# -*- coding: utf-8 -*-

import torch
# from scipy.optimize import fmin_ncg
from scipy.optimize import minimize
from copy import deepcopy


class InfluenceFunction():
    def __init__(self, trainloader, **kwargs):
        self.sample_size = 64
        self.module = self._model.classifier[0]
        self.H_inv = None

    def calc_v(self, z_test, z_test_label):
        self._model.zero_grad()

        X_var = deepcopy(z_test)
        scores = self.model(X_var)
        loss = self.criterion(scores, z_test_label)

        grad = torch.autograd.grad(loss, self.parameter, retain_graph=True)[
            0].view(-1).cuda().detach()
        return grad

    def calc_H(self):
        H_sum = None
        for i, (Xi, Yi) in enumerate(self.trainloader):
            if i >= self.sample_size:
                break
            X = to_tensor(Xi)
            Y = to_tensor(Yi, dtype='long')
            H = compute_hessian((self._model.get_final_fm(X), Y), self.module)
            if H_sum is None:
                H_sum = H
            else:
                H_sum += H
        H_sum /= self.sample_size
        return H_sum

    def calc_Ht(self, t):
        X = torch.Tensor([])
        Y = torch.LongTensor([])
        self._model.zero_grad()
        for i, (Xi, Yi) in enumerate(self.trainloader):
            X = to_tensor(Xi)
            Y = to_tensor(Yi, dtype='long')
            break
        X.requires_grad = True
        loss = self.criterion(self.model(X), Y)
        grad = torch.autograd.grad(
            loss, self.parameter, create_graph=True)[0].view(-1)
        grad_v = torch.dot(grad, t.detach())
        result = torch.autograd.grad(
            grad_v, self.parameter, retain_graph=True)[0].view(-1)
        return result.detach()

    def get_f_fn(self):
        def f_fn(x):
            _x = to_tensor(x)
            Hx = self.calc_Ht(_x)
            result = torch.dot(0.5 * Hx - self.v, _x)
            print(1)
            return to_numpy(result)
        return f_fn

    def get_grad_fn(self):
        def grad_fn(x):
            _x = to_tensor(x)
            Hx = self.calc_Ht(_x)
            result = Hx - self.v
            print(2)
            return to_numpy(result)
        return grad_fn

    def get_fhess_p_fn(self):
        def fhess_p_fn(x, p):
            _p = to_tensor(p)
            Hp = self.calc_Ht(_p)
            print(3)
            print(_p.norm(p=2))
            return to_numpy(Hp)
        return fhess_p_fn

    def calc_s_test(self, cg=False):
        if not cg:
            return self.H_inv.mm(self.v.view(-1, 1)).view(-1)
        f_fn = self.get_f_fn()
        grad_fn = self.get_grad_fn()
        fhess_p_fn = self.get_fhess_p_fn()
        fmin_results = minimize(f_fn, x0=to_numpy(self.v), method='Newton-CG', jac=grad_fn, hessp=fhess_p_fn, options={
                                'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'maxiter': 1, 'disp': False, 'return_all': False})
        # fmin_results = fmin_ncg(
        #     f=f_fn,
        #     x0=to_numpy(self.v),
        #     fprime=grad_fn,
        #     fhess_p=fhess_p_fn,
        #     avextol=1e-8,
        #     maxiter=1)
        return to_tensor(fmin_results).detach()

    def up_loss(self, z, z_label, parameter, cg=False, H_inv=None):
        self.H_inv = H_inv
        self.parameter = parameter

        self.v = self.calc_v(z, z_label)
        s_test = self.calc_s_test(cg=cg)
        return max(1e-10, float(torch.dot(self.v, s_test).item()))


def compute_hessian(batch, module):
    features, labels = batch
    N, D, C = len(features), module.in_features, module.out_features
    logits = module(features)
    probs = F.softmax(logits, dim=-1)
    labels_onehot = torch.zeros(N, C, device='cuda')
    labels_onehot.scatter_(1, labels[:, None], 1)
    Hess = [[] for _ in range(C)]
    HHT = torch.matmul(features.view(N, D, 1), features.view(N, 1, D))
    for i in range(C):
        for j in range(C):
            h = torch.zeros(D, D, device='cuda')
            yi, yj = probs[:, i, None, None], probs[:, j, None, None]
            yi, yj = yi, yj
            h = h - yi * yj * HHT
            if i == j:
                h = h + yi * HHT
            Hess[i].append(h.mean(0))
        Hess[i] = torch.stack(Hess[i], dim=0)
    Hess = torch.stack(Hess, dim=0)
    Hess = Hess.permute(0, 2, 1, 3)
    Hess = Hess.contiguous().view(C * D, C * D)
    return Hess


def compute_gradient(batch, module):
    features, labels = batch
    N, D, C = len(features), module.in_features, module.out_features
    logits = module(features)
    probs = F.softmax(logits, dim=-1)
    labels_onehot = torch.zeros(N, C, device=env['device'])
    labels_onehot.scatter_(1, labels[:, None], 1)
    diff = probs - labels_onehot
    weights_grad = []
    for c in range(C):
        weights_grad.append((diff[:, c].unsqueeze(-1) * features).sum(0))
    weights_grad = torch.stack(weights_grad, 0)
    return weights_grad
