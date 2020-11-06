# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    name = 'Defense Performance'
    # fig.set_axis_label('x', 'Defense')
    # fig.set_axis_label('y', 'Attack')
    # fig.set_axis_label('z', 'Defense Performance')

    _fig = plt.figure(figsize=(10, 7.5))
    _ax = _fig.add_subplot(projection='3d')
    fig = Figure(name, fig=_fig, ax=_ax)
    ax = fig.ax

    color_list = [ting_color['red_carrot'], ting_color['blue'], ting_color['yellow']]

    defense_list = ['NC', 'TABOR', 'STRIP', 'NEO', 'AT', 'MagNet']
    attack_list = ['BN', 'LB', 'TNN', 'IMC', 'RB', 'TB', 'ESB', 'ABE']
    data = {
        'group1': {
            attack_list[0]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[1]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[2]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[3]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[4]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[5]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[6]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[7]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
        },
        'group2': {
            attack_list[0]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[1]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[2]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[3]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[4]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[5]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[6]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[7]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
        },
        'group3': {
            attack_list[0]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[1]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[2]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[3]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[4]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[5]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[6]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
            attack_list[7]: [96.078, 96.078, 96.078, 96.078, 95.146, 94.118],
        },
    }

    defense_mesh, attack_mesh = np.meshgrid(defense_list, attack_list)
    defense_mesh, attack_mesh = defense_mesh.ravel(), attack_mesh.ravel()
    defense_idx = {v: k for k, v in enumerate(defense_list)}
    attack_idx = {v: k for k, v in enumerate(attack_list)}
    defense_pos = np.array([defense_idx[defense] for defense in defense_mesh])
    attack_pos = np.array([attack_idx[attack] for attack in attack_mesh])

    for i, (group, sub_data) in enumerate(list(data.items())):
        z_list = np.array([sub_data[attack][defense_idx[defense]]
                           for attack, defense in zip(attack_mesh, defense_mesh)])
        fig.bar3d(defense_pos + (i - 1.5) / 4, attack_pos, z_list, size=0.5 / 4, color=color_list[i], label=group)
    fig.set_axis_lim(axis='y', lim=[0.0, 8.0], margin=[0.5, 0.5], piece=len(defense_list))
    fig.set_axis_lim(axis='x', lim=[0.0, 8.0], margin=[0.5, 0.5], piece=len(attack_list))
    fig.set_axis_lim(axis='z', lim=[0.0, 100.0], margin=[0, 3], piece=5)
    fig.set_axis_label('y', 'Defense')
    fig.set_axis_label('x', 'Attack')
    fig.set_axis_label('z', 'Defense Performance')

    ax.set_xticklabels(defense_list, rotation=0)
    ax.set_yticklabels(attack_list, rotation=0)

    fig.set_title()
    # fig.set_legend()

    plt.show()
