# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    name = 'figure Attack Optimization Term Study'
    fig = Figure(name)
    fig.set_axis_label('x', r'Trigger Transparency ($\mathbf{\alpha}$)')
    fig.set_axis_label('y', 'ASR (%)')
    fig.set_axis_lim('x', lim=[0, 0.9], piece=9, margin=[0.05, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[40, 100], piece=3, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title('')

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
        # if key != 'trojannn':
        #     continue

        y_list = np.array(y[key])
        x_list = np.array(x[:len(y_list)])
        x_grid = np.linspace(0.0, 0.9, 9000)
        y_grid = np.linspace(0.0, 0.9, 9000)

        if key in ['imc', 'latent_backdoor']:
            y_grid = fig.interp_fit(x_list, y_list, x_grid)
            y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
            y_grid = fig.monotone(y_grid, increase=False)
            y_grid = fig.avg_smooth(y_grid, window=100)
            y_grid = fig.avg_smooth(y_grid, window=200)
            y_grid = fig.avg_smooth(y_grid, window=300)
            y_grid = fig.avg_smooth(y_grid, window=400)
        if key in ['badnet', 'trojannn']:
            y_grid = fig.exp_fit(x_list, y_list, x_grid, degree=5, increase=False, epsilon=5)
            y_grid = fig.monotone(y_grid, increase=False)

        fig.curve(x_grid, y_grid, color=color_dict[key])
        fig.scatter(x_list, y_list, color=color_dict[key], marker=mark_dict[key], label=attack_mapping[key])

        # fig.curve(x[:len(value)], value, color=color_dict[key], marker=mark_dict[key], label=attack_mapping[key])
        # fig.scatter(x[:len(value)], value, color=color_dict[key], marker=mark_dict[key])
    fig.set_legend(ncol=1, labelspacing=0.2, loc='lower left')
    fig.save(folder_path='./result/')
