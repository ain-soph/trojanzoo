# -*- coding: utf-8 -*-

import numpy as np

x = np.linspace(0, 1, 6)
y = np.linspace(0, 1, 6)

p00 = -0.3018
p10 = 6623
p01 = 0.1794
p20 = -8.295e+06
p11 =  -681
p02 = -0.007589
p30 = 0
p21 = 6.483e+05
p12 = 13.33
p03 = 0.0001031
p40 = 0
p31 = 0
p22 = 0
p13 = -0
p04 = -0


def f(x, y):
    return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3 + p04*y**4


for _x in x:
    for _y in y:
        print('%.1f' % _x, '%.1f' % _y, '%.1f' % (f(_x, _y)*100))
