# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    simple_parser = argparse.ArgumentParser()
    simple_parser.add_argument('--group_name', dest='group_name', default="model", type=str)
    args, unknown = simple_parser.parse_known_args()

    name = 'Defense Performance'
    # fig.set_axis_label('x', 'Defense')
    # fig.set_axis_label('y', 'Attack')
    # fig.set_axis_label('z', 'Defense Performance')

    _fig = plt.figure(figsize=(10, 7.5))
    _ax = _fig.add_subplot(projection='3d')
    fig = Figure(name, fig=_fig, ax=_ax)
    ax = fig.ax

    color_list = [ting_color['red_carrot'], ting_color['blue'], ting_color['yellow'], ting_color['green']]

    # defense_list = ['NC', 'TABOR', 'STRIP', 'NEO', 'AT', 'MagNet']
    # attack_list = ['BN', 'LB', 'TNN', 'IMC', 'RB', 'TB', 'ESB', 'ABE']
    defense_list = ['DI', 'IT', 'MagNet', 'neo', 'neuron_inspect', 'strip']
    attack_list = ['BN', 'BE', 'IMC', 'LB', 'RB', 'TN', 'TNN']
    data = {"dataset": {
        # 'cifar10': {
        #     attack_list[0]: [0, 0, 0, 0, 0, 0],
        #     attack_list[1]: [0, 0, 0, 0, 0, 0],
        #     attack_list[2]: [0, 0, 0, 0, 0, 0],
        #     attack_list[3]: [0, 0, 0, 0, 0, 0],
        #     attack_list[4]: [0, 0, 0, 0, 0, 0],
        #     attack_list[5]: [0, 0, 0, 0, 0, 0],
        #     attack_list[6]: [0, 0, 0, 0, 0, 0],
        #     attack_list[7]: [0, 0, 0, 0, 0, 0],
        # },
        # 'sample_vggface2': {
        #     attack_list[0]: [0, 0, 0, 0, 0, 0],
        #     attack_list[1]: [0, 0, 0, 0, 0, 0],
        #     attack_list[2]: [0, 0, 0, 0, 0, 0],
        #     attack_list[3]: [0, 0, 0, 0, 0, 0],
        #     attack_list[4]: [0, 0, 0, 0, 0, 0],
        #     attack_list[5]: [0, 0, 0, 0, 0, 0],
        #     attack_list[6]: [0, 0, 0, 0, 0, 0],
        #     attack_list[7]: [0, 0, 0, 0, 0, 0],
        # },
        # 'sample_imagenet': {
        #     attack_list[0]: [0, 0, 0, 0, 0, 0],
        #     attack_list[1]: [0, 0, 0, 0, 0, 0],
        #     attack_list[2]: [0, 0, 0, 0, 0, 0],
        #     attack_list[3]: [0, 0, 0, 0, 0, 0],
        #     attack_list[4]: [0, 0, 0, 0, 0, 0],
        #     attack_list[5]: [0, 0, 0, 0, 0, 0],
        #     attack_list[6]: [0, 0, 0, 0, 0, 0],
        #     attack_list[7]: [0, 0, 0, 0, 0, 0],
        # },
        # 'gtsrb': {
        #     attack_list[0]: [0, 0, 0, 0, 0, 0],
        #     attack_list[1]: [0, 0, 0, 0, 0, 0],
        #     attack_list[2]: [0, 0, 0, 0, 0, 0],
        #     attack_list[3]: [0, 0, 0, 0, 0, 0],
        #     attack_list[4]: [0, 0, 0, 0, 0, 0],
        #     attack_list[5]: [0, 0, 0, 0, 0, 0],
        #     attack_list[6]: [0, 0, 0, 0, 0, 0],
        #     attack_list[7]: [0, 0, 0, 0, 0, 0],
        # },
    },
        "model": {
            # 'resnetcomp18': {
            #     attack_list[0]: [0, 0, 0, 0, 0, 0],
            #     attack_list[1]: [0, 0, 0, 0, 0, 0],
            #     attack_list[2]: [0, 0, 0, 0, 0, 0],
            #     attack_list[3]: [0, 0, 0, 0, 0, 0],
            #     attack_list[4]: [0, 0, 0, 0, 0, 0],
            #     attack_list[5]: [0, 0, 0, 0, 0, 0],
            #     attack_list[6]: [0, 0, 0, 0, 0, 0],
            #     attack_list[7]: [0, 0, 0, 0, 0, 0],
            # },
            # 'vggcomp13': {
            #     attack_list[0]: [0, 0, 0, 0, 0, 0],
            #     attack_list[1]: [0, 0, 0, 0, 0, 0],
            #     attack_list[2]: [0, 0, 0, 0, 0, 0],
            #     attack_list[3]: [0, 0, 0, 0, 0, 0],
            #     attack_list[4]: [0, 0, 0, 0, 0, 0],
            #     attack_list[5]: [0, 0, 0, 0, 0, 0],
            #     attack_list[6]: [0, 0, 0, 0, 0, 0],
            #     attack_list[7]: [0, 0, 0, 0, 0, 0],
            # },
            # 'densecomp121': {
            #     attack_list[0]: [0, 0, 0, 0, 0, 0],
            #     attack_list[1]: [0, 0, 0, 0, 0, 0],
            #     attack_list[2]: [0, 0, 0, 0, 0, 0],
            #     attack_list[3]: [0, 0, 0, 0, 0, 0],
            #     attack_list[4]: [0, 0, 0, 0, 0, 0],
            #     attack_list[5]: [0, 0, 0, 0, 0, 0],
            #     attack_list[6]: [0, 0, 0, 0, 0, 0],
            #     attack_list[7]: [0, 0, 0, 0, 0, 0],
            # }
        },
        "trigger": {
            # defense_list = ['NC', 'TABOR', 'STRIP', 'NEO', 'AT', 'MagNet']
            # attack_list = ['BN', 'LB', 'TNN', 'IMC', 'RB', 'TB', 'ESB', 'ABE']
            # defense_list_curr = ['DI', 'IT', 'MagNet', 'neo', 'neuron_inspect', 'strip']
            # attack_list_curr = ['BN', 'BE', 'IMC', 'LB', 'RB', 'TN', 'TNN']
            'as': {
                'BN': [6.5694, 2.18, 6.01, 0.29, 0.1686, 0.0643],
                'BE': [5.6031, 0.53, 4.62, 0.28, 0.3373, 0.0737],
                'IMC': [3.9812, 0.22, 16.03, 0.29, 0.6744, 0.9914],
                'LB': [2.2931, 4.14, 42.64, 0.29, 0.0, 0.9159],
                'RB': [2.4445, 5.42, 78.62, 0.29, 3.3725, 0.3435],
                'TN': [0, 89.164, 0.0, 1.3586, 0.0288, 0], # missing deep inspect and magnet
                'TNN': [3.4686, 0.43, 37.4, 0.23, 0.2698, 0.1318]
            },
            'aS': {
                'BN': [0.9539, 21.3, 5.75, 0.44, 0.6745, 0.2626],
                'BE': [1.5145, 15.22, 4.64, 0.47, 0.0, 0.0849],
                'IMC': [1.7453, 63.61, 80.21, 0.79, 3.5973, 0.9602],
                'LB': [0.6745, 19.98, 43.04, 0.74, 0.6745, 0.9863],
                'RB': [2.0526, 29.2, 87.01, 0.56, 0.4047, 0.0351],
                'TN': [0.0674, 0.23, 1.95, 0.02, 1.349, 0.116],
                'TNN': [0.9015, 8.08, 87.07, 0.52, 0.6745, 0.03]
            },
            'As': {
                'BN': [2.4544, 12.92, 41.81, 0.07, 1.0118, 0.1511],
                'BE': [2.7102, 14.45, 39.25, 0.13, 1.3489, 0.1583],
                'IMC': [4.8077, 1.48, 88.34, 0.17, 0.1124, 0.9132],
                'LB': [3.7307, 3.65, 88.59, 0.29, 2.4728, 0.4268],
                'RB': [8.3637, 26.38, 66.33, 0.1, 0.5621, 0.106],
                'TN': [0.0572, 0.08, 2.42, 0.04, 1.4839, 0.0946],
                'TNN': [7.4782, 7.96, 70.48, 0.1, 0.2248, 0.1426]
            },
            'AS': {
                'BN': [2.7031, 61.28, 40.97, 0.52, 0.8994, 0.2519],
                'BE': [6.8866, 47.36, 35.68, 0.51, 1.3491, 0.2397],
                'IMC': [0.6969, 28.75, 87.9, 0.65, 0.6745, 0.9382],
                'LB': [0.8822, 32.68, 87.97, 0.7, 2.5293, 0.9001],
                'RB': [3.1342, 64.68, 77.41, 0.59, 1.3489, 0.1766],
                'TN': [0.0674, 0.06, 2.38, 0.02, 1.349, 0.0958],
                'TNN': [4.4833, 70.71, 73.4, 0.49, 0.6745, 0.2039]
            }
        },


    }

    data = data[args.group_name]
    defense_mesh, attack_mesh = np.meshgrid(defense_list, attack_list)
    defense_mesh, attack_mesh = defense_mesh.ravel(), attack_mesh.ravel()
    defense_idx = {v: k for k, v in enumerate(defense_list)}
    attack_idx = {v: k for k, v in enumerate(attack_list)}
    defense_pos = np.array([defense_idx[defense] for defense in defense_mesh])
    attack_pos = np.array([attack_idx[attack] for attack in attack_mesh])

    for i, (group, sub_data) in enumerate(list(data.items())):
        z_list = np.array([sub_data[attack][defense_idx[defense]]
                           for attack, defense in zip(attack_mesh, defense_mesh)])
        fig.bar3d(defense_pos + i / 8 - 0.25, attack_pos - 1/16, z_list, size=0.5 / 4, color=color_list[i], label=group, shade=True)
    fig.set_axis_lim(axis='x', lim=[0.0, 6.0], margin=[0.2, 0.2], piece=len(defense_list))
    fig.set_axis_lim(axis='y', lim=[0.0, 7.0], margin=[0.2, 0.2], piece=len(attack_list))
    fig.set_axis_lim(axis='z', lim=[0.0, 80.0], margin=[0, 3], piece=4)
    fig.set_axis_label('x', 'Defense')
    fig.set_axis_label('y', 'Attack')
    fig.set_axis_label('z', 'Defense Performance')

    ax.set_xticklabels(defense_list, rotation=0)
    ax.set_yticklabels(attack_list, rotation=0)

    fig.set_title()
    # fig.set_legend()

    plt.show()
