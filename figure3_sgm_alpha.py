# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    name = 'figure3 SGM alpha'
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
        'badnet': {
            'original': [85.060, 81.820, 78.940, 77.070, 72.990, 66.160, 56.580, 43.540, 30.320, 21.120],
            'sgm': [76.530, 74.670, 72.540, 69.340, 64.670, 58.230, 48.730],
        },
    }
    for i, (attack, sub_dict) in enumerate(y.items()):
        for j, (mode, value) in enumerate(sub_dict.items()):
            fig.curve(x[:len(value)], value, color=color_list[i + j], label=f'{attack} {mode}')
            fig.scatter(x[:len(value)], value, color=color_list[i + j])
    fig.save('./result/')
