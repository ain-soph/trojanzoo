#!/usr/bin/env python3

from trojanvision.utils.model_archs.darts import Genotype
from copy import deepcopy

org_arch = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])


def mitigation_1(arch: Genotype) -> Genotype:
    '''make arch deep'''
    arch = deepcopy(arch)
    for cell in [arch.normal, arch.reduce]:
        for i, (op, idx) in enumerate(cell):
            if i in [0, 1]:
                cell[i] = (op, i)
            else:
                cell[i] = (op, i // 2 + 1)
    return arch


def mitigation_2(arch: Genotype) -> Genotype:
    '''replace skip connect to sep conv'''
    arch = deepcopy(arch)
    for cell in [arch.normal, arch.reduce]:
        for i, (op, idx) in enumerate(cell):
            if op == 'skip_connect':
                cell[i] = ('sep_conv_3x3', idx)
    return arch


def mitigation_3(arch: Genotype) -> Genotype:
    return mitigation_2(mitigation_1(arch))


print('DARTS-i  : ', mitigation_1(org_arch))
print('DARTS-ii : ', mitigation_2(org_arch))
print('DARTS-iii: ', mitigation_3(org_arch))
