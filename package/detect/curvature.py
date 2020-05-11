# -*- coding: utf-8 -*-
from package.imports.universal import *
from package.utils.utils import *


class Curvature():
    def __init__(self, model=None, h=0.1, **kwargs):
        self.h = h
        self.model = model

    def compute_gradient(self, x, y):
        x = x.detach()
        x.requires_grad = True
        loss = self.model.loss(x, y)
        grad = torch.autograd.grad(loss, x)[0]
        x = x.detach()
        return grad

    def measure(self, x, y, d=None):
        x = x.detach()
        n = len(x)
        gx1 = self.compute_gradient(x, y)
        if d is None:
            d = gx1.sign()
            d = d.view(n, -1)
            d = d / torch.norm(d, dim=-1, keepdim=True)
            d = d.view(*x.shape)
        x2 = x + self.h * d
        gx2 = self.compute_gradient(x2, y)
        diff = (gx2 - gx1)
        diff = diff.view(n, -1)
        return (diff * diff).sum(1)

    def benign_measure(self, validloader=None, batch_num=10):
        if validloader is None:
            validloader = self.model.dataset.loader['valid']
        measure_list = []
        for i, (x, y) in enumerate(validloader):
            if i >= batch_num:
                break
            measure = self.measure(x, y)
            measure_list.append(float(measure.mean()))
        return np.mean(measure_list)
