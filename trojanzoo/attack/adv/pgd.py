# -*- coding: utf-8 -*-
from ..attack import Attack

from trojanzoo.utils import add_noise
from trojanzoo.utils.output import prints, output_memory

import torch
from typing import Union, List
from collections.abc import Callable


class PGD(Attack):
    r"""Projected Gradient Descent.
    Args:
        alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.
    """

    name = 'pgd'

    def __init__(self, alpha: float = 3.0 / 255, epsilon: float = 8.0 / 255, targeted=True,
                 universal=False, norm=float('inf'), mode: str = 'white', sigma=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.param_list['pgd'] = ['alpha', 'epsilon', 'targeted', 'norm', 'mode', 'sigma']

        self.alpha = alpha
        self.epsilon = epsilon

        self.targeted = targeted
        self.universal = universal
        self.norm = norm

        self.mode = mode
        self.sigma = sigma

    def attack(self, _input: torch.Tensor, noise: torch.Tensor = None,
               iteration: int = None, alpha: float = None, epsilon: float = None,
               loss_fn: Callable = None, target: Union[torch.LongTensor, int] = None,
               output: Union[int, List[str]] = None, indent: int = None, **kwargs):
        # ------------------------------ Parameter Initialization ---------------------------------- #

        if iteration is None:
            iteration = self.iteration
        if alpha is None:
            alpha = self.alpha
        if epsilon is None:
            epsilon = self.epsilon
        if indent is None:
            indent = self.indent
        output = self.get_output(output)

        if loss_fn is None:
            if target is None:
                assert not self.targeted
                target = self.model.get_class(_input)
            elif isinstance(target, int):
                target = torch.ones_like(_input) * target

            def _loss_fn(_X):
                loss = self.model.loss(_X, target, **kwargs)
                return loss if targeted else -loss
            loss_fn = _loss_fn
        # ----------------------------------------------------------------------------------------- #

        if iteration == 0 or alpha == 0.0 or epsilon == 0.0:
            return _input, None

        if 'init' in output:
            self.output_info(noise=noise, _input=_input, indent=indent, mode='init', loss_fn=loss_fn)
        if self.model:
            self.model.eval()

        if noise is None:
            noise = torch.zeros_like(_input[0] if self.universal else _input)
        X = add_noise(_input, noise, batch=self.universal)

        # ----------------------------------------------------------------------------------------- #

        for _iter in range(iteration):
            if self.early_stop:
                _confidence, _classification = self.model.get_prob(X).max(dim=1)
                if (self.targeted and _classification.equal(target) and _confidence.min() > self.stop_confidence) or \
                        (not self.targeted and (_classification - target).abs().min() > 0):
                    if 'final' in output:
                        self.output_info(noise=noise, _input=X, indent=indent, mode='final', loss_fn=loss_fn)
                    return X, _iter + 1
            if self.mode == 'white':
                X.requires_grad = True
                loss = loss_fn(X)
                grad = torch.autograd.grad(loss, X)[0]
                X.requires_grad = False
            elif self.mode == 'black':
                grad = self.cal_gradient(
                    loss_fn, X, sigma=self.sigma)
                if 'middle' in output:
                    X.requires_grad = True
                    loss = loss_fn(X)
                    real_grad = torch.autograd.grad(loss, X)[0].detach()
                    X.requires_grad = False
                    prints('cos<real, est> = ', self.cos_sim(grad.sign(), real_grad),
                           indent=indent + 2)
            else:
                raise NotImplementedError()
            if self.universal:
                grad = grad.mean(dim=0)
            noise.data = (noise - alpha * torch.sign(grad)).data
            noise.data = self.projector(noise, epsilon, norm=self.norm).data
            X = add_noise(_input, noise, batch=self.universal)
            if self.universal:
                noise_new = (X - _input).data
                noise.data = (noise_new.sign() * noise_new.abs().min(dim=0)).data
            else:
                noise.data = (X - _input).data

            if 'middle' in output:
                self.output_info(noise=noise, _input=_input, indent=indent, mode='middle',
                                 _iter=_iter, iteration=iteration, loss_fn=loss_fn)

        if 'final' in output:
            self.output_info(noise=noise, _input=_input, indent=indent, mode='final', loss_fn=loss_fn)
        return X, None

    @staticmethod
    def cos_sim(a, b):
        return (a * b).sum() / a.norm(p=2) / b.norm(p=2)

    def output_info(self, noise: torch.Tensor, _input: torch.Tensor, indent=None, mode='init', _iter=0, iteration=0, loss_fn=None):
        if indent is None:
            indent = self.indent
        # if _result is None:
        #     assert _input is not None
        #     self.model.eval()
        #     _result = self.model.get_prob(_input)
        # _confidence, _classification = _result.max(1)

        if mode in ['init', 'final']:
            prints('{name} Attack {mode} Classification'.format(name=self.name, mode=mode), indent=indent)
        elif mode in ['middle']:
            indent += 4
            self.output_iter(name=self.name, _iter=_iter, iteration=iteration, indent=indent)
        with torch.no_grad():
            loss = float(loss_fn(_input + noise))
            norm = noise.norm(p=self.norm)
            prints('L-{p} norm: {norm}    loss: {loss:.5f}'.format(p=self.norm, norm=norm, loss=loss))

        # for i in range(len(_input)):
        #     prefix = 'idx: %d  %s: ' % (i, 'Max')
        #     prints('{0:<20s} {klass:<4d} {confidence:.5f}'.format(prefix, int(_classification[i]), float(_confidence[i])),
        #            indent=indent + 2)
        #     prefix = 'idx: %d  %s: ' % (i, 'Target' if self.targeted else 'Untarget')
        #     prints('{0:<20s} {klass:<4d} {confidence:.5f}'.format(prefix, int(target[i]), float(_result[i][target[i]])),
        #            indent=indent + 2)
        if 'memory' in self.output:
            output_memory(indent=indent + 4)
