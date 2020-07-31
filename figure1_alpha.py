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

    color_list = [ting_color['red'], ting_color['yellow'], ting_color['green'], ting_color['blue']]

    x = np.linspace(0.0, 1.0, 11)
    y = {
        'cifar10': {
            'badnet': [78.710, 76.910, 73.940, 67.310, 61.980, 54.450, 45.330, 37.290, 29.610, 22.960],
            'latent_backdoor': [97.160, 95.450, 90.080, 86.490, 82.010, 71.660, 55.440, 23.150, 10.190, 10.280],
            'trojannn': [75.360, 73.530, 71.450, 68.090, 64.060, 56.350, 48.030, 38.770, 30.910, 12.260],
        },
        'gtsrb': {
            'badnet': [56.757, 58.540, 56.888, 51.952, 44.651, 40.165, 39.02, 31.325, 28.341, 23.104, 3.247],
            'latent_backdoor': [92.080, 84.666, 84.234, 80.556, 78.491, 72.823, 63.382, 36.074, 0.976, 0.713, 0.713],
            'trojannn': [59.816, 57.902, 55.405, 52.196, 48.217, 41.667, 42.492, 35.098, 27.721, 20.702, 3.453],
            'targeted_backdoor': [33.821, 23.592, 30.950, 26.051, 27.44, 20.383, 16.967, 15.447, 2.928, 2.628, 2.477],
        },
    }
    for i, (key, value) in enumerate(y[args.dataset].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')
