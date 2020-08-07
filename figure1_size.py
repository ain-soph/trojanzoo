# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    args = parser.parse_args()
    name = 'figure1 %s size' % args.dataset
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 10], piece=10, margin=[0, 0.5],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['red_deep'], ting_color['yellow'], 
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'], ting_color['green'], ]

    x = np.linspace(1, 10, 10)
    y = {
        'cifar10': {
            'badnet': [67.240, 78.710, 85.060, 88.150, 90.590, 91.550, 93.230],
            'latent_backdoor': [31.890, 97.160, 99.000, 99.790, 100.000, 100.000, 100.000],
            'trojannn': [63.690, 75.360, 96.400, 96.190, 93.190, 86.280, 76.070],
            'imc': [19.610, 99.900, 98.410],
            'clean_label_pgd': [52.080, 68.790, 76.290, 78.760, 81.960, 85.740, 90.040],
        },
        'gtsrb': {
            'badnet': [42.68, 56.757, 53.053, 59.791, 56.044, 55.03, 61.787, 73.724, 76.107, 79.936],
            'latent_backdoor': [9.572, 92.080, 99.981, 99.944, 100, 100, 100, 100, 100, 100],
            'trojannn': [37.125, 59.816, 57.789, 58.146, 73.048, 94.125, 98.104, 84.441, 93.919, 95.420],
            'targeted_backdoor': [23.161, 40.165, 40.897, 41.498, 40.034, 46.021, 50.544, 55.011, 61.787, 60.773],
        },
    }
    for i, (key, value) in enumerate(y[args.dataset].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')
