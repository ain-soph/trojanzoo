# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    parser.add_argument('--high', dest='high', action='store_true')
    args = parser.parse_args()
    name = 'figure sgm %s' % args.dataset
    if args.high:
        name += ' high'
    fig = Figure(name)
    fig.set_axis_label('x', 'Attack Name')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'], ting_color['green'], color['brown']['brown']]

    y = {
        'benign': {
            'badnet': 72.381,
            'latent': 100.000,
            'trojannn': 91.509,
            'imc': 77.940,
        },
        'sgm': {
            'badnet': 57.370,
            'latent': 98.660,
            'trojannn': 45.830,
            'imc': 80.940,
        },
    }

    z = {
        'benign': {
            'badnet': 79.380,
            'latent': 100.000,
            'trojannn': 92.360,
            'imc': 91.050,
        },
        'sgm': {
            'badnet': 61.320,
            'latent': 99.770,
            'trojannn': 81.480,
            'imc': 93.000,
        },
    }
    y = z if args.high else y
    x = np.linspace(0.2, 0.8, len(list(y['benign'].keys())))

    x_list = list(y['benign'].keys())
    y_list = [(y['sgm'][key] - y['benign'][key]) for key in x_list]
    fig.bar(x, y_list, width=0.05, color=color_list[0])

    fig.set_axis_lim('x', lim=[0, 1.0], piece=len(x_list) + 1, margin=[0.05, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[-50, 50], piece=6,
                     _format='%.1f')
    x_list.append(None)
    x_list.insert(0, None)
    fig.ax.set_xticklabels(x_list)
    fig.save('./result/')
