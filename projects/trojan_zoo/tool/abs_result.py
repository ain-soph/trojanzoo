import numpy as np
from trojanzoo.utils import *

a = np.load('/data/rbp5354/result/cifar10/resnetcomp18/abs/badnetrandom_pos_square_white_tar0_alpha0.00_mark(3,3)_best.npy', allow_pickle=True).item()

values = [a[i]['attack_loss'] + 0.07 * a[i]['norm'] for i in a.keys()]
normalize_mad(values)

a[0]['norm']
a[0]['jaccard']


a = np.load('/data/rbp5354/result/cifar10/resnetcomp18/mitigate/badnetsquare_white_tar0_alpha0.00_mark(3,3)_best.npy',
            allow_pickle=True).item()

values = [a[i]['attack_loss'] + 0.07 * a[i]['norm'] for i in a.keys()]
normalize_mad(values)

a[0]['norm']
a[0]['jaccard']
