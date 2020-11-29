# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    name = 'figure Attack Optimization Term Study'
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Transparency')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=10, margin=[0.05, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[40, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    mark_dict = {
        'badnet': 'H',
        'trojannn': '^',
        'reflection_backdoor': 'o',
        'targeted_backdoor': 'v',
        'latent_backdoor': 's',
        'trojannet': 'p',
        'bypass_embed': 'h',
        'imc': 'D',
    }
    color_dict = {
        'badnet': ting_color['red_carrot'],
        'trojannn': ting_color['green'],
        'reflection_backdoor': ting_color['blue'],
        'targeted_backdoor': ting_color['yellow'],
        'latent_backdoor': ting_color['red_deep'],
        'trojannet': ting_color['purple'],
        'bypass_embed': ting_color['blue_light'],
        'imc': color['brown']['brown'],
    }
    attack_mapping = {
        'badnet': 'BN',
        'trojannn': 'TNN',
        'reflection_backdoor': 'RB',
        'targeted_backdoor': 'TB',
        'latent_backdoor': 'LB',
        'trojannet': 'ESB',
        'bypass_embed': 'ABE',
        'imc': 'IMC',
    }
    x = np.linspace(0.0, 1.0, 11)
    y = {
        'badnet': [94.220, 93.860, 93.380, 92.020, 90.370, 88.030, 86.180, 81.270, 73.680, 49.580],
        'imc': [100.000, 100.000, 100.000, 100.000, 99.980, 99.960, 99.900, 99.910, 97.900, 93.630],
        'trojannn': [99.710, 99.720, 99.740, 99.770, 99.820, 99.710, 99.060, 96.070, 88.870, 65.550],
        'latent_backdoor': [100.000, 100.000, 100.000, 100.000, 100.000, 99.990, 99.880, 99.410, 97.730, 95.370],
    }
    for i, (key, value) in enumerate(y.items()):
        fig.curve(x[:len(value)], value, color=color_dict[key], marker=mark_dict[key], label=attack_mapping[key])
        fig.scatter(x[:len(value)], value, color=color_dict[key], marker=mark_dict[key])
    fig.set_legend()
    fig.save(folder_path='./result/')
