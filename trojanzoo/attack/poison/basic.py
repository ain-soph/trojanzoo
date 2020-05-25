# -*- coding: utf-8 -*-
from .perturb import Perturb
from package.utils.utils import *
from package.utils.output import prints, output_iter
from package.imports.universal import *

import math
import random


class Poison(Perturb):

    name = 'poison'

    def __init__(self, iteration=None, epoch=None, poison_percent=None, poison_num=None, lr=0.01, full=True, train_opt='partial', optim_type='Adam', lr_scheduler=False, early_stop=False, **kwargs):
        super().__init__(iteration=iteration, early_stop=early_stop, **kwargs)

        self.poison_percent = poison_percent
        self.poison_num = poison_num
        assert poison_percent is None or poison_num is None
        assert poison_percent is not None or poison_num is not None
        self.lr = lr
        self.epoch = epoch
        self.full = full

        self.optimizer = self.model.define_optimizer(
            train_opt=train_opt, optim_type=optim_type, lr=self.lr, lr_scheduler=lr_scheduler)
        if lr_scheduler:
            self.lr_scheduler = self.optimizer
            self.optimizer = self.lr_scheduler.optimizer

        self.output_par()

    def get_output_int(self, org_output=0):
        if org_output < 1:
            return set()
        elif org_output < 5:
            return set('param')
        elif org_output < 10:
            return set(['param', 'final'])
        elif org_output < 20:
            return set(['param', 'init', 'final'])
        elif org_output < 22:
            return set(['param', 'init', 'final', 'middle'])
        elif org_output < 24:
            return set(['param', 'init', 'final', 'middle', 'epoch'])
        elif org_output < 26:
            return set(['param', 'init', 'final', 'middle', 'epoch', 'validate'])
        elif org_output < 28:
            return set(['param', 'init', 'final', 'middle', 'iter', 'validate'])
        elif org_output < 30:
            return set(['param', 'init', 'final', 'middle', 'iter', 'validate_iter'])
        else:
            return set(['init', 'final', 'middle', 'iter', 'validate_iter', 'memory'])

    def perturb(self, X_var, target, iteration=None, epoch=None, poison_percent=None, poison_num=None, output=None, indent=None, early_stop=None, stop_confidence=None, full=None, **kwargs):
        if iteration is None:
            iteration = self.iteration
        if epoch is None:
            epoch = self.epoch
        length = len(self.model.dataset.loader['train'])
        if iteration is None:
            assert epoch is not None
            iteration = epoch * length
        elif epoch is not None:
            iteration += epoch * length
        if early_stop is None:
            early_stop = early_stop
        if stop_confidence is None:
            stop_confidence = self.stop_confidence
        if output is None:
            output = self.output
        if indent is None:
            indent = self.indent

        if full is None:
            full = self.full
        loader = self.model.dataset.loader['train'] if full else self.model.dataset.loader['train2']

        self.output_result(target=target, _input=X_var,
                           output=output, indent=indent)
        if poison_percent is None:
            poison_percent = self.poison_percent
            if poison_num is None:
                poison_num = self.poison_num
                if 0.0 in [poison_num, poison_percent]:
                    return None
            else:
                assert self.poison_percent is None
                if poison_num == 0.0:
                    return None
        else:
            assert poison_num is None
            if poison_percent == 0.0:
                return None

        if poison_percent is None:
            poison_percent = self.poison_percent
        if poison_num is None:
            poison_num = self.poison_num
        if poison_num is not None:
            if poison_num == 0:
                return None
        else:
            assert poison_percent is not None
            if poison_percent == 0.0:
                return None

        self.optimizer.zero_grad()
        _iter = 0
        while True:
            if _iter >= iteration:
                break
            if early_stop:
                _confidence, _classification = self.model.get_prob(X_var).max(1)
                if _classification.equal(target) and _confidence.min() > self.stop_confidence:
                    self.output_result(target=target, _input=X_var,
                                       output=output, indent=indent, mode='final')
                    if epoch is None:
                        return _iter
                    else:
                        return int(_iter / length)+1
            for data in loader:
                if _iter >= iteration:
                    break
                _input, _label = self.model.dataset.get_data(data)
                batch_size = len(_label)
                if poison_num is None:
                    poison_num = batch_size*poison_percent
                if poison_num >= 1:
                    poison_num = int(poison_num)
                elif random.random() < poison_num:
                    poison_num = 1
                else:
                    poison_num = 0
                if poison_num > 0:
                    batch_input = repeat_to_batch(
                        to_tensor(X_var[0]), poison_num)
                    batch_target = repeat_to_batch(
                        to_tensor(target).squeeze(), poison_num)
                    _input = torch.cat((_input, batch_input)).detach()
                    _label = torch.cat((_label, batch_target)).detach()
                self.model.train()

                # data = [batch_input, batch_target]
                # _input, _output, _label = get_logits(data, model)
                # zero the gradients buffer, reset the gradient in each update
                loss = self.model.loss(_input, _label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.model.eval()
                if 'middle' in output and 'iter' in output:
                    self.output_middle(target=target, _input=X_var,
                                       _iter=_iter, iteration=iteration, length=length,
                                       output=output, indent=indent)
                if 'middle' in output and 'validate_iter' in output:
                    self.model._validate(full=False, indent=indent+5)

                empty_cache(self.model.cache_threshold)
                _iter += 1
            if 'middle' in output and 'epoch' in output:
                self.output_middle(target=target, _input=X_var,
                                   _iter=_iter, iteration=iteration, length=length,
                                   output=output, indent=indent)
            if 'middle' in output and 'validate' in output:
                self.model._validate(full=False, indent=indent+5)

        self.output_result(target=target, _input=X_var,
                           output=output, indent=indent, mode='final')
        return None

    def output_iter(self, name=None, _iter=0, iteration=None, length=None, output=None, indent=0):
        if output is None:
            output = self.output
        if name is None:
            name = self.name
        if length is None:
            length = len(self.model.dataset.loader['train'])

        string = name
        if 'epoch' in output:
            _epoch = int(_iter / length)
            epoch = math.ceil(float(iteration)/length)
            string += ' Epoch: ' + output_iter(_epoch, iteration=epoch)
            if 'iter' in output:
                string += ' Iter: ' + output_iter(
                    _iter % length+1, min(iteration-_epoch*length, length))
        elif 'iter' in output:
            string += ' Iter: ' + output_iter(_iter, iteration)
        else:
            return
        prints(string, indent=indent)
