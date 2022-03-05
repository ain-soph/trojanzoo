#!/usr/bin/env python3

from trojanvision.models import ImageModel
import torch


class Curvature():
    # TODO: finish the detect()
    def __init__(self, model: ImageModel = None, h: float = 0.1, **kwargs):
        self.h: float = h
        self.model: ImageModel = model

    def compute_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach()
        x.requires_grad = True
        loss = self.model.loss(x, y)
        grad = torch.autograd.grad(loss, x)[0]
        return grad

    def measure(self, x: torch.Tensor, y: torch.Tensor, d: torch.Tensor = None) -> torch.Tensor:
        gx1 = self.compute_gradient(x, y)
        if d is None:
            d = gx1.sign().flatten(start_dim=1)
            d /= torch.norm(d, dim=-1, keepdim=True)
            d = d.view_as(x)
        x2 = x + self.h * d
        gx2 = self.compute_gradient(x2, y)
        diff = (gx2 - gx1).flatten(start_dim=1)
        return (diff * diff).sum(1)

    def benign_measure(self, validloader=None, batch_num=20):
        if validloader is None:
            validloader = self.model.dataset.loader['valid']
        measure_list = []
        for i, data in enumerate(validloader):
            _input, _label = self.model.get_data(data)
            if i >= batch_num:
                break
            measure = self.measure(_input, _label)
            measure_list.extend(measure.detach().cpu().tolist())
        return measure_list
