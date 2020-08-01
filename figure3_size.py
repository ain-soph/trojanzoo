# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack', dest='attack', default='badnet')
    args = parser.parse_args()
    name = 'figure3 %s size' % args.attack
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 50], piece=10, margin=[0, 5.0],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['yellow'], ting_color['green'], ting_color['blue']]

    x = np.linspace(1.0, 7.0, 7)**2
    y = {
        'badnet': {
            'vgg': [63.580, 76.590, 85.110, 88.020, 90.510, 91.600, 92.420],
            'resnet': [67.240, 78.710, 85.060, 88.150, 90.590, 91.550, 93.230],
            'densenet': [67.250, 78.700, 81.210, 88.140, 90.600, 91.540, 93.260],
        },
        'latent_backdoor': {
            'vgg': [30.620, 93.570, 99.820, 99.990, 99.950, 99.910, 99.920],
            'resnet': [31.890, 97.160, 99.000, 99.790, 100.000, 100.000, 100.000],
            'densenet': [31.930, 97.130, 99.660, 99.330, 100.000, 99.980, 95.130],
        },
        'trojannn': {
            'vgg': [59.450, 75.410, 83.880, 96.590],  # , 88.270, 79.000
            'resnet': [63.690, 75.360, 82.730, 96.190],
            'densenet': [63.700, 75.390, 82.790, 96.210],  # , 65.810, 37.640
        },
    }
    for i, (key, value) in enumerate(y[args.attack].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')
