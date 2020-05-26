# -*- coding: utf-8 -*-
from .perturb import Perturb
from .pgd import PGD
from .poison import Poison
from package.utils.utils import *
from package.imports.universal import *


class Unify(Perturb):

    name = 'unify'

    def __init__(self, param={'pgd': {}, 'poison': {}}, iteration=20, output=0, indent=0, **kwargs):
        super().__init__(iteration=iteration, output=output, **kwargs)

        for module in param.values():
            if 'model' not in module.keys() and 'model' in kwargs:
                module['model'] = kwargs['model']
            if 'stop_confidence' not in module.keys() and 'stop_confidence' in kwargs:
                module['stop_confidence'] = kwargs['stop_confidence']

            if 'indent' not in module.keys():
                module['indent'] = self.indent+8
            if isinstance(output, int):
                output_child = output-30
                module['output'] = output_child
        self.module.pgd = PGD(**param['pgd'])
        self.module.poison = Poison(**param['poison'])

        self.output_par()

    def perturb(self, _input, target=None, param={'pgd': {}, 'poison': {}}, iteration=None, output=None, indent=None, early_stop=None, stop_confidence=None, **kwargs):
        output = self.get_output(output)
        if indent is None:
            indent = self.indent
        else:
            param['pgd']['indent'] = indent+self.module.pgd.indent-self.indent
            param['poison']['indent'] = indent + \
                self.module.poison.indent-self.indent
        if iteration is None:
            iteration = self.iteration
        if early_stop is None:
            early_stop = self.early_stop
        if stop_confidence is None:
            stop_confidence = self.stop_confidence

        self.output_result(_input=_input, target=target,
                           output=output, indent=indent)

        adv_X = _input
        noise = to_tensor(torch.zeros_like(_input))
        for _iter in range(iteration):
            if early_stop:
                _confidence, _classification = self.model.get_prob(adv_X.detach()).max(1)
                if _classification.equal(target) and _confidence.min() > self.stop_confidence:
                    self.output_result(target=target, _input=adv_X,
                                       output=output, indent=indent, mode='final')
                    if 'final' in output:
                        self.model._validate(full=False)
                    return adv_X, _iter + 1
            adv_X, pgd_iter = self.module.pgd.perturb(_input, noise=noise, target=target, targeted=True,
                                                      **(param['pgd']))

            self.model.load_pretrained_weights()

            poison_iter = self.module.poison.perturb(adv_X, target,
                                                     **(param['poison']))

            self.output_middle(target=target, _input=adv_X, _iter=_iter, iteration=iteration,
                               output=output, indent=indent)
        self.output_result(target=target, _input=adv_X,
                           output=output, indent=indent, mode='final')
        if 'final' in output:
            self.model._validate(full=False)
        return adv_X, None
