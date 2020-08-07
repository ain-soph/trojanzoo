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
    name = 'figure1 %s alpha' % args.dataset
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Transparency')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=10, margin=[0.05, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'], ting_color['green'], ]

    x = np.linspace(0.0, 1.0, 11)
    y = {
        'cifar10': {
            'badnet': [85.060, 81.820, 78.940, 77.070, 72.990, 66.160, 56.580, 43.540, 30.320, 21.120],
            'latent_backdoor': [99.000, 99.810, 99.760, 99.300, 98.940, 98.310, 96.420, 91.290, 72.430, 13.960],
            'trojannn': [96.400, 95.200, 93.300, 90.000, 83.950, 74.020, 60.390, 44.300, 28.820, 21.200],
            'imc': [98.410, 100.000, 99.980, 99.720, 93.980, 88.240, 73.110],
            'clean_label_pgd': [76.290, 69.330, 61.770, 49.310, 34.600, 19.110, 12.990, 11.640, 11.420, 11.280]
        },
        'gtsrb': {
            'badnet': [53.153, 52.965, 52.046, 49.869, 46.539, 41.929, 37.782, 32.057, 28.679, 22.710, 3.247],
            # 'latent_backdoor': [92.080, 84.666, 84.234, 80.556, 78.491, 72.823, 63.382, 36.074, 0.976, 0.713, 0.713],
            'trojannn': [53.041, 46.059, 45.083, 42.361, 40.484, 38.514, 35.83, 31.119, 25.563, 18.975, 2.665],
            'targeted_backdoor': [40.897, 34.553, 31.794, 25.882, 19.651, 11.355, 6.55, 4.523, 3.397, 2.947, 2.721],
        },
    }
    # y = {
    #     'cifar10': {
    #         'badnet': [78.710, 76.910, 73.940, 67.310, 61.980, 54.450, 45.330, 37.290, 29.610, 22.960],
    #         'latent_backdoor': [97.160, 95.450, 90.080, 86.490, 82.010, 71.660, 55.440, 23.150, 10.190, 10.280],
    #         'trojannn': [75.360, 73.530, 71.450, 68.090, 64.060, 56.350, 48.030, 38.770, 30.910, 12.260],
    #     },
    #     'gtsrb': {
    #         'badnet': [56.757, 58.540, 56.813, 51.971, 44.613, 40.128, 39.02, 31.344, 28.378, 23.104, 3.247],
    #         'latent_backdoor': [92.080, 84.666, 84.234, 80.556, 78.491, 72.823, 63.382, 36.074, 0.976, 0.713, 0.713],
    #         'trojannn': [59.816, 58.84, 56.175, 53.416, 49.906, 43.431, 42.492, 35.098, 27.721, 23.292, 3.453],
    #         'targeted_backdoor': [40.165, 33.315, 30.950, 26.051, 27.44, 20.383, 16.967, 15.447, 2.928, 2.628, 2.477],
    #     },
    # }
    for i, (key, value) in enumerate(y[args.dataset].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')
