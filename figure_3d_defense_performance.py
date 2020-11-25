# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    simple_parser = argparse.ArgumentParser()
    simple_parser.add_argument('--group_name', dest='group_name', default="trigger", type=str)
    simple_parser.add_argument('--defense_group', dest='defense_group', default="inference", type=str)
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

    defense_mapping = {
        'general': {
            'recompress': 'DU',
            'randomized_smooth': 'RS',
            'adv_train': 'AR',
            'magnet': 'MP',
            'fine_pruning': 'FP',
        },
        'model_inspection': {
            'neural_cleanse': 'NC',
            'deep_inspect': 'DI',
            'tabor': 'TABOR',
            'neuron_inspect': 'NI',
            'ABS': 'ABS',
        },
        'inference': {
            'strip': 'STRIP',
            'neo': 'NEO',
        }
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
    file_path = './defense_3d_data.json'
    with open(file_path) as f:
        data = json.load(f)
    data = data[args.group_name]
    defense_mapping = defense_mapping[args.defense_group]

    defense_list = list(defense_mapping.keys())
    attack_list = list(attack_mapping.keys())

    defense_mesh, attack_mesh = np.meshgrid(defense_list, attack_list)
    defense_mesh, attack_mesh = defense_mesh.ravel(), attack_mesh.ravel()
    defense_idx = {v: k for k, v in enumerate(defense_list)}
    attack_idx = {v: k for k, v in enumerate(attack_list)}
    defense_pos = np.array([defense_idx[defense] for defense in defense_mesh])
    attack_pos = np.array([attack_idx[attack] for attack in attack_mesh])

    for i, (group, sub_data) in enumerate(list(data.items())):
        z_list = np.array([sub_data[attack][defense]
                           for attack, defense in zip(attack_mesh, defense_mesh)])

        if args.group_name == 'trigger':
            attack_values = attack_pos
            defense_values = defense_pos
            print(group)
            if group == 'aS':
                attack_values = attack_pos + i / 3
            elif group == 'As':
                defense_values = defense_pos + i / 32
            elif group == 'AS':
                attack_values = attack_pos + i / 9
                defense_values = defense_pos + i / 48
            fig.bar3d(attack_values - 1 / 3, defense_values - 1 / 16, z_list,
                      size=(1 / 3, 1 / 16), color=color_list[i], label=group, shade=True)
        else:
            fig.bar3d(attack_pos + i / 8 - 0.25, defense_pos - 1 / 16, z_list,
                      size=(1 / 3, 1 / 16), color=color_list[i], label=group, shade=True)
    fig.set_axis_lim(axis='y', lim=[0.0, len(defense_list) - 1], margin=[0.2, 0.2], piece=len(defense_list) - 1)
    fig.set_axis_lim(axis='x', lim=[0.0, len(attack_list) - 1], margin=[0.2, 0.2], piece=len(attack_list) - 1)
    fig.set_axis_lim(axis='z', lim=[0.0, 100.0], margin=[0, 3], piece=5)
    fig.set_axis_label('y', 'Defense')
    fig.set_axis_label('x', 'Attack')
    fig.set_axis_label('z', 'Defense Performance')

    ax.set_yticklabels([defense_mapping[defense] for defense in defense_list], rotation=0)
    ax.set_xticklabels([attack_mapping[attack] for attack in attack_list], rotation=0)

    fig.set_title()
    # fig.set_legend()

    plt.show()
