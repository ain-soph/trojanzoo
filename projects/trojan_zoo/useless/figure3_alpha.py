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
    name = 'figure3 %s alpha' % args.attack
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Transparency')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=10, margin=[0.05, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['yellow'], ting_color['green'], ting_color['blue']]

    x = np.linspace(0.0, 1.0, 11)
    y = {
        'badnet': {
            'vgg': [76.590, 74.800, 72.300, 68.550, 63.550, 56.780, 47.930, 37.240, 29.670, 22.210],
            'resnet': [78.710, 76.910, 73.940, 67.310, 61.980, 54.450, 45.330, 37.290, 29.610, 22.960],
            'densenet': [78.700, 76.830, 72.530, 67.280, 61.920, 54.530, 45.360, 37.270, 29.560, 23.000],
        },
        'latent_backdoor': {
            'vgg': [93.570, 93.000, 87.370, 82.170, 80.040, 69.360, 52.560, 20.410, 10.230, 10.310],
            'resnet': [97.160, 95.450, 90.080, 86.490, 82.010, 71.660, 55.440, 23.150, 10.190, 10.280],
            'densenet': [97.130, 95.480, 90.110, 86.580, 82.020, 71.600, 55.400, 23.180, 10.210, 10.280],
        },
        'trojannn': {
            'vgg': [75.410, 73.110, 69.740, 65.610, 59.730, 55.400, 45.030, 37.780, 29.370, 22.350],
            'resnet': [75.360, 73.530, 71.450, 68.090, 64.060, 56.350, 48.030, 38.770, 30.910, 12.260],
            'densenet': [75.390, 73.530, 71.480, 67.960, 63.940, 56.300, 48.020, 38.840, 30.880, 12.280],
        },
    }
    for i, (key, value) in enumerate(y[args.attack].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')
