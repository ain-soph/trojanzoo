# -*- coding: utf-8 -*-

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from trojanzoo.plot import *

import numpy as np

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    fig = Figure('auc')
    ax = fig.ax
    fig.set_axis_label('y', 'ASR')
    fig.set_axis_label('x', 'Clean Accuracy Drop')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=5, margin=[0.02, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 1.0], piece=5, margin=[0.0, 0.05],
                     _format='%.1f')
    fig.set_title('')

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
    attack_list = ['badnet', 'trojannn', 'reflection_backdoor', 'targeted_backdoor',
                   'latent_backdoor', 'bypass_embed', 'imc', ]
    line_list = []
    label_list = []
    for i, attack in enumerate(attack_list):
        key = attack
        label = attack_mapping[attack]
        _dict = np.load(f'./result/auc/{attack}.npy', allow_pickle=True).item()
        label_list.append((label, f'{_dict["auc"]:.3f}'))
        line = fig.curve(_dict['x_grid'], _dict['y_grid'], color=color_dict[key])
        line_list.append(fig.curve_legend(color=color_dict[key], marker=mark_dict[key]))

        fig.scatter(_dict['x'], _dict['y'], color=color_dict[key], marker=mark_dict[key])

    x1 = np.linspace(0, 1, 100)
    y1 = x1
    fig.curve(x=x1, y=y1, color=ting_color["grey"], linewidth=5, linestyle='--')
    empty_list = []
    for i in range(len(line_list)):
        line, = fig.ax.plot([], linestyle='None', color='k')
        empty_list.append(line)
    label_list = [i[0] for i in label_list] + [i[1] for i in label_list]
    fig.set_legend(line_list + empty_list, label_list, ncol=2, columnspacing=-2.0, fontsize=12)

    fig.save(folder_path='./result/auc/')
