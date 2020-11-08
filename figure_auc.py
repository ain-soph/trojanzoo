# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    fig = Figure(name='auc')
    ax = fig.ax
    fig.set_axis_label('x', 'Attack Success Rate')
    fig.set_axis_label('y', 'Clean Accuracy Drop')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=5, margin=[0.0, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 1.0], piece=5, margin=[0.0, 0.05],
                     _format='%.1f')

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'],
                  ting_color['green'], color['brown']['brown'], color['green']['army']]
    mark_list = ['H', '<', 'o', 'v', 's', 'p', '*', 'h', 'D']
    fig.set_title()

    attack_list = ['badnet', 'latent_backdoor', 'trojannn', 'imc',
                   'reflection_backdoor', 'targeted_backdoor', 'clean_label', 'bypass_embed', ]
    for i, attack in enumerate(attack_list):
        _dict = np.load(f'./result/auc/{attack}.npy', allow_pickle=True).item()
        fig.curve(_dict['x_grid'], _dict['y_grid'], color=color_list[i], label=f'{attack} auc {_dict["auc"]:.3f}')
        fig.scatter(_dict['x'], _dict['y'], color=color_list[i], marker=mark_list[i])

    x1 = np.linspace(0, 1, 100)
    y1 = x1
    fig.curve(x=x1, y=y1, color=ting_color["grey"], linewidth=5, linestyle='--')

    fig.save("./result/auc/")
