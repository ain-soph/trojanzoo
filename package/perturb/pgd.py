# -*- coding: utf-8 -*-

from .perturb import Perturb

from package.imports.universal import *
from package.utils.utils import add_noise, to_tensor
from package.utils.output import prints


class PGD(Perturb):
    """Projected Gradient Descent, ref: https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py"""

    name = 'pgd'

    def __init__(self, alpha=3.0/255, epsilon=8.0/255, targeted=True, p=float('inf'), mode='white', sigma=1e-3, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.p = p
        self.mode = mode

        self.sigma = sigma

        self.output_par()

    @staticmethod
    def cos_sim(a, b):
        a = to_tensor(a)
        b = to_tensor(b)
        return (a*b).sum()/a.norm(p=2)/b.norm(p=2)

    def perturb(self, _input, target=None, targeted=None, noise=None, iteration=None, alpha=-1.0, epsilon=-1.0, mode=None, p=None, batch=False, early_stop=None, stop_confidence=None, loss_func=None, output=None, indent=None, **kwargs):
        #------------------------------ Parameter Initialization ----------------------------------#
        if targeted is None:
            targeted = self.targeted
        if target is None:
            if targeted:
                raise ValueError()
            else:
                target = self.model.get_class(_input)

        # elif targeted is None:
        #     _classification = self.model.get_class(_input)
        #     if (_classification-target).abs().bool().float().mean() > 0.8:
        #         targeted = True
        #     else:
        #         targeted = False

        if early_stop is None:
            early_stop = self.early_stop
        if stop_confidence is None:
            stop_confidence = self.stop_confidence
        output = self.get_output(output)
        if indent is None:
            indent = self.indent
        if iteration is None:
            iteration = self.iteration
        if alpha == -1.0:
            alpha = self.alpha
        if epsilon == -1.0:
            epsilon = self.epsilon
        if p is None:
            p = self.p
        if mode is None:
            mode=self.mode

        def _loss_func(_X):
            loss = self.model.loss(_X, target, **kwargs)
            return loss if targeted else -loss
        if loss_func is None:
            loss_func=_loss_func
        #-----------------------------------------------------------------------------------------#

        if iteration == 0 or alpha == 0.0 or epsilon == 0.0:
            return _input, None

        self.output_result(target=target, targeted=targeted, _input=_input,
                           output=output, indent=indent)
        self.model.eval()


        if noise is None:
            if batch:
                noise = to_tensor(torch.zeros_like(_input[0])).detach()
            else:
                noise = to_tensor(torch.zeros_like(_input)).detach()
        X = add_noise(_input, noise, batch=batch)

        #-----------------------------------------------------------------------------------------#

        for _iter in range(iteration):

            if early_stop:
                _confidence, _classification = self.model.get_prob(
                    X).max(dim=1)
                if targeted:
                    if _classification.equal(target) and _confidence.min() > stop_confidence:
                        self.output_result(target=target, targeted=targeted, _input=X,
                                           output=output, indent=indent, mode='final')
                        return X, _iter + 1
                else:
                    if (_classification-target).abs().min() > 0:
                        self.output_result(target=target, targeted=targeted, _input=X,
                                           output=output, indent=indent, mode='final')
                        return X, _iter + 1

            if self.mode == 'white':
                X.requires_grad = True
                self.model.zero_grad()
                loss = loss_func(X)
                grad = torch.autograd.grad(loss, X)[0].detach()
            elif self.mode == 'black':
                grad = self.cal_gradient(
                    loss_func, X, sigma=self.sigma).detach()
                if 'middle' in output:
                    X.requires_grad = True
                    self.model.zero_grad()
                    loss = loss_func(X)
                    real_grad = torch.autograd.grad(loss, X)[0].detach()
                    prints('cos<real, est> = ', self.cos_sim(grad.sign(), real_grad.sign()),
                           indent=indent+2)
            else:
                raise ValueError(
                    'Value of Parameter "mode" should be "white" or "black"!')
            if batch:
                grad = grad.mean(dim=0)
            noise.data = (noise - alpha * torch.sign(grad)).data
            X = add_noise(_input, noise, batch=batch)

            self.projector(noise, epsilon, p=p)
            self.output_middle(target=target, targeted=targeted, _input=X, _iter=_iter, iteration=iteration,
                               output=output, indent=indent)

        self.output_result(target=target, targeted=targeted, _input=X,
                           output=output, indent=indent, mode='final')
        return X, None
