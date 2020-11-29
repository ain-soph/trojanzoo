# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np
import json

import seaborn as sns

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

    fig = Figure(name, figsize=(20, 5))
    ax = fig.ax
    ax.set_aspect('equal')

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

    matrix = np.zeros([len(defense_list) * (len(data.keys()) + 1), len(attack_list) * 2])
    if args.group_name == 'trigger':
        matrix = np.zeros([len(defense_list) * 3, len(attack_list) * 3])

    for i, (group, sub_data) in enumerate(list(data.items())):
        z_list = np.array([sub_data[attack][defense]
                           for attack, defense in zip(attack_mesh, defense_mesh)])
        offset = [0, 0]
        if args.group_name == 'trigger':
            if group == 'As':
                offset = [0, 0]
            elif group == 'AS':
                offset = [1, 0]
            elif group == 'as':
                offset = [0, 1]
            elif group == 'aS':
                offset = [1, 1]
            for y, x, z in zip(attack_pos, defense_pos, z_list):
                matrix[x * 3 + offset[0], y * 3 + offset[1]] = z
        else:
            for y, x, z in zip(attack_pos, defense_pos, z_list):
                matrix[x * (len(data.keys()) + 1) + i, y * 2] = z
    sns.heatmap(matrix, annot=True, ax=fig.ax, cmap='coolwarm', fmt='.2f', linewidths=1)
    if args.group_name == 'trigger':
        fig.set_axis_lim(axis='y', lim=[1, 1 + 3 * (len(defense_list) - 1)], margin=[1, 1], piece=len(defense_list) - 1)
        fig.set_axis_lim(axis='x', lim=[1, 1 + 3 * (len(attack_list) - 1)], margin=[1, 1], piece=len(attack_list) - 1)
    else:
        fig.set_axis_lim(axis='y', lim=[1, 4], margin=[1, 1], piece=len(defense_list) - 1)
        fig.set_axis_lim(axis='x', lim=[1, 22], margin=[1, 1], piece=len(attack_list) - 1)
    fig.set_axis_label('y', 'Defense')
    fig.set_axis_label('x', 'Attack')

    ax.set_yticklabels([defense_mapping[defense] for defense in defense_list], rotation=0)
    ax.set_xticklabels([attack_mapping[attack] for attack in attack_list], rotation=0)

    # fig.set_title()
    # fig.set_legend()

    plt.show()
    # fig.save('./result.png')
