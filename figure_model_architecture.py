# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    name = "Model Architecture"
    fig = Figure(name)
    fig.set_axis_label('x', 'Attack')
    fig.set_axis_label('y', 'Attack Successful Rate')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'], ting_color['green'],
                  color['brown']['brown']]

    data_source = {
        'badnet': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'trojannn': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'hidden_trigger': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'latent_backdoor': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'reflection_backdoor': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'clean_label': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'bypass_embed': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'imc': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        },
        'trojannet': {
            'resnetcomp18': 95.0,
            'densenetcomp121': 96.0,
            'vggcomp13': 97.0
        }
    }

    x = np.linspace(0, 1, len(data_source.keys()) * 3)
    x_list = [item for attack in data_source.keys() for item in data_source[attack].keys()]
    x_label = [[attack, "", ""] for attack in data_source.keys()]
    x_label = [i for sublist in x_label for i in sublist]
    y_list = [item for attack in data_source.keys() for item in data_source[attack].values()]

    fig.bar(x, y_list, width=0.03, color=color_list[0])
    fig.set_axis_lim('x', lim=[0, 1.0], piece=len(x_list) + 1, margin=[0.00, 0.00],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 100], piece=10,
                     _format='%.1f')
    fig.ax.set_xticklabels(x_label)
    plt.show()
