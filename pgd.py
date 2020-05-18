# -*- coding: utf-8 -*-

from package.imports.universal import *
from package.utils.utils import *
from package.parse.perturb.pgd import Parser_PGD


import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------ #

parser_pgd = Parser_PGD()
dataset = parser_pgd.module.dataset
model = parser_pgd.module.model
perturb = parser_pgd.module.perturb

testloader = dataset.loader['train']

# model._validate()
correct = 0
total = 0
total_iter = 0
for i, data in enumerate(testloader):
    if i > 100:
        break
    _input, _label = model.remove_misclassify_from_batch(data)
    if len(_label) == 0:
        continue
    target = perturb.generate_target(_input) if perturb.targeted else _label
    adv_input, _iter = perturb.perturb(
        _input, target=target)
    total += 1
    if _iter is not None:
        correct += 1
        total_iter += _iter
    print(i)
    print((adv_input-_input).abs().max())
    print(_iter)
    print('succ rate: ', float(correct)/total)
    if correct > 0:
        print('avg  iter: ', float(total_iter)/correct)
    print()
