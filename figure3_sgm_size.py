# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    name = 'figure3 SGM size'
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 7], piece=7, margin=[0, 0.5],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'], ting_color['green'], ]

    x = np.linspace(1, 10, 10)
    y = {
        # 'badnet': {
        #     'original': [67.240, 78.710, 85.060, 88.150, 90.590, 91.550, 93.230],
        #     'sgm': [59.190, 68.580, 76.530, 79.080, 81.750, 83.670],
        # },
        'imc': {
            'original': [67.240, 78.710, 85.060, 88.150, 90.590, 91.550, 93.230],
            'sgm': [59.190, 68.580, 76.530, 79.080, 81.750, 83.670],
        },
    }
    for i, (attack, sub_dict) in enumerate(y.items()):
        for j, (mode, value) in enumerate(sub_dict.items()):
            fig.curve(x[:len(value)], value, color=color_list[i + j], label=f'{attack} {mode}')
            fig.scatter(x[:len(value)], value, color=color_list[i + j])
    fig.save('./result/')
