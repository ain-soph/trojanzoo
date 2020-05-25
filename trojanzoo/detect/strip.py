# -*- coding: utf-8 -*-

from package.imports.universal import *

from package.utils.utils import *


class STRIP():
    def __init__(self, model, alpha=0.5, boundary=0.05, N=8, detach=True):
        self.model = model
        self.trainloader = self.model.dataset.loader['train']
        self.trainloader2 = self.model.dataset.get_dataloader('train', batch_size=1)
        self.alpha = alpha
        self.boundary = boundary
        self.N = N
        self.detach = True

    def superimpose(self, _input1, _input2, alpha=None):
        if alpha is None:
            alpha = self.alpha
        _input2 = _input2[:_input1.shape[0]]

        result = alpha*_input1+(1-alpha)*_input2
        if self.detach:
            result = result.detach()
        return result

    def entropy(self, _input):
        p = self.model.softmax(self.model(_input))
        return (-p*torch.log(p)).sum(1).mean()

    def detect(self, _input):
        h = 0.0
        for i, data in enumerate(self.trainloader):
            if i >= self.N:
                break
            X, Y = self.model.get_data(data, mode='train')
            _test = self.superimpose(_input, X)
            entropy = self.entropy(_test)
            h += entropy
        h /= self.N
        if h < self.boundary:
            return True, h
        return False, h

    def detect_label(self, _input, label):
        batch_num = _input.shape[0]
        for i, data in enumerate(self.trainloader2):
            X, Y = self.model.get_data(data, mode='train')
            if Y == label:
                tensor_list = repeat_to_batch(X[0], batch_num)
        X = to_tensor(tensor_list)
        _test = self.superimpose(_input, X)
        entropy = self.entropy(_test)
        return entropy

    def detect_multi(self, _input, num_classes):
        h = 0.0
        for label in range(num_classes):
            h += self.detect_label(_input, label)
        return h/num_classes

    def detect_new(self, _input, nc_mask, nc_trigger, _perturb):
        h_list = []
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            h = 0.0
            counter = 0
            for i, (X, Y) in enumerate(self.trainloader):
                if i >= self.N:
                    break
                X = to_tensor(X)
                X_mark = _perturb.add_mark(X, nc_trigger, nc_mask)
                _test = self.superimpose(_input, X_mark, alpha=alpha)
                entropy = float(self.entropy(_test))
                h += entropy
                counter += 1
            h /= counter
            h_list.append(h)
        return np.std(h_list)
