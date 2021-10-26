#!/usr/bin/env python3

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# https://github.com/quark0/darts
# https://github.com/automl/RobustDARTS
nasnet = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ],
    reduce_concat=[4, 5, 6],
)
nasnet_adapt = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ],
    reduce_concat=[4, 5, 6],
)

amoebanet = Genotype(
    normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ],
    normal_concat=[4, 5, 6],
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 2), ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('conv_7x1_1x7', 0), ('sep_conv_3x3', 5), ],
    reduce_concat=[3, 4, 6]
)
amoebanet_adapt = Genotype(
    normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ],
    normal_concat=[4, 5, 6],
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_7x7', 1), ('sep_conv_7x7', 0),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('conv_7x1_1x7', 0), ('sep_conv_3x3', 1), ],
    reduce_concat=[3, 4, 6]
)

darts_v1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
darts_v2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])
darts = darts_v2

snas_mild = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])

snas_adapt = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

enas = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1),
            ('avg_pool_3x3', 1), ('sep_conv_5x5', 4), ('avg_pool_3x3', 1), ('sep_conv_3x3', 5), ('sep_conv_5x5', 0)],
    reduce_concat=[2, 3, 6])

enas_adapt = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 0), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
    reduce_concat=[2, 3, 6])

robust_darts = Genotype(
    normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0),
            ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
    reduce_concat=range(2, 6))

# https://github.com/chenxin061/pdarts
pdarts = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
            ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)],
    reduce_concat=range(2, 6))

# https://github.com/yuhuixu1993/PC-DARTS
pc_darts_cifar = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
    reduce_concat=range(2, 6))
pc_darts_image = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)],
    reduce_concat=range(2, 6))
pc_darts = pc_darts_cifar

# https://github.com/xiangning-chen/DrNAS
drnas_cifar10 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
            ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('skip_connect', 4), ('sep_conv_5x5', 1)],
    reduce_concat=range(2, 6))
drnas_imagenet = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)],
    reduce_concat=range(2, 6))
drnas = drnas_cifar10

# https://github.com/lightaime/sgas
sgas = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_5x5', 2)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2),
            ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
    reduce_concat=range(2, 6))


diy_deep = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 3), ('skip_connect', 3), ('skip_connect', 4), ('dil_conv_3x3', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

diy_noskip = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

diy_deep_noskip = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4), ('dil_conv_3x3', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

random = Genotype(
    normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 1),
            ('skip_connect', 3), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)],
    reduce_concat=[2, 3, 4, 5])
