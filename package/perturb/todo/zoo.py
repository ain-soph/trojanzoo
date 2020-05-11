# -*- coding: utf-8 -*-
from perturb.perturb import *

class ZOO(Perturb):
    """Projected Gradient Descent, ref: https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py"""

    def __init__(self, iteration=20, output=None, **kwargs):
        super(ZOO, self).__init__(iteration=iteration, output=output)
        self.name = 'ZOO'

    def zoo_adam(self, f, X, step_size, e, i, h, M, v, T, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        gi = (f(X+h*e)-f(X-h*e))/2*h
        # hi = (f(X+h*e)-2*f(X)+f(X-h*e))/(h*h)
        T[i] += 1

        M[i] = beta_1*M[i] + (1-beta_1) * gi
        v[i] = beta_2*v[i] + (1-beta_2) * gi

        M[i] = M[i]/(1-beta_1**T[i])
        v[i] = v[i]/(1-beta_2**T[i])

        delta = -step_size*M[i]/(v[i]**0.5+epsilon)
        return delta

    def zoo_newton(self, f, X, step_size, e, i, h, **kwargs):
        gi = (f(X+h*e)-f(X-h*e))/2*h
        hi = (f(X+h*e)-2*f(X)+f(X-h*e))/(h*h)

        delta = -step_size*gi
        if hi > 0:
            delta /= hi
        return delta

    # Stochastic Coordinate Descent
    # X0 shape: batch_size, channels, height, width
    def SCD(self, f, X0, step_size, h=0.0001, mode='adam', **kwargs):
        X = to_tensor(deepcopy(X0.detach()))

        M = torch.zeros_like(X[0].view(-1))
        v = torch.zeros_like(X[0].view(-1))
        T = torch.zeros_like(X[0].view(-1))

        for _iter in range(self.iteration):
            e = torch.zeros_like(X[0].view(-1))
            i = randint(0, e.shape[0])
            e[i] = 1
            e = e.view(X[0].shape)
            e = repeat_to_batch(e, batch_size=X.shape[0])

            delta = eval('self.zoo_'+mode)(f, X, step_size, e, i,
                                           h, M=M, v=v, T=T, **kwargs)
            X += delta * e