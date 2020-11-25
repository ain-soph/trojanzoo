# -*- coding: utf-8 -*-, from trojanzoo.plot import *

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    args = parser.parse_args()
    name = 'figure1 %s size' % args.dataset
    fig = Figure(name, figsize=(5, 3))
    fig.set_axis_label('x', r'Trigger size ($|\mathit{m}|$)')
    fig.set_axis_label('y', 'Attack success rate (%)')
    fig.set_axis_lim('x', lim=[2, 5], piece=3, margin=[0.5, 0.5],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
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
    y = {
        'cifar10': {
            'badnet': [61.520, 70.520, 72.381, 77.350, 79.380, 81.040, 81.560],
            'latent_backdoor': [10.720, 99.250, 100.000, 100.000, 100.000, 100.000, 100.000],
            'trojannn': [46.600, 87.770, 91.509, 91.910, 92.360, 93.520, 94.990],
            'imc': [58.550, 99.660, 99.960, 99.990, 100.000, 100.000, 100.000],
            'reflection_backdoor': [44.560, 64.300, 79.240, 88.150, 92.920, 94.000, 96.390],
            'targeted_backdoor': [10.940, 11.140, 11.470, 11.760, 33.290, 44.450, 49.000],
            # 'clean_label_pgd': [12.190, 12.410, 12.650, 13.040, 13.240, 13.030, 14.650],
            'trojannet': [10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352],
            'bypass_embed': [66.700, 74.270, 74.320, 78.520, 83.340, 83.650, 85.610],
        },
        'gtsrb': {
            'badnet': [0.619, 61.543, 65.634, 71.415, 71.772, 71.753, 72.954, 71.565, 73.949, 75],
            'latent_backdoor': [99.625, 99.23, 98.423, 99.249, 99.662, 99.887, 99.925, 99.887, 99.925, 99.962],
            'trojannn': [0.601, 57.508, 71.697, 69.67, 72.11, 73.011, 78.96, 81.963, 82.658, 83.483],
            'imc': [21.34, 92.399, 97.579, 95.89, 96.509, 98.986, 99.095, 98.874, 98.911, 98.968],
            'reflection_backdoor': [3.003, 38.589, 42.774, 48.311, 53.848, 62.218, 64.492, 74.437, 72.879, 85.511],
            'targeted_backdoor': [0.619, 0.619, 0.601, 0.619, 0.638, 0.601, 0.601, 0.788, 0.807, 0.77],
            # 'clean_label_pgd': [1.858, 1.464, 0.938, 1.745, 0.601, 1.014, 0.582, 1.839, 1.276, 0.807],
            'trojannet': [0.582, 0.582, 0.582, 0.582, 0.582, 0.582, 0.582, 0.563],
            'bypass_embed': [7.432, 61.974, 68.412, 73.78, 73.142, 73.104, 74.474, 76.52, 79.279, 78.829],
        },
        'sample_imagenet': {
            'badnet': [11.400, 83.400, 89.800, 91.200, 91.400, 91.400, 91.400],
            'latent_backdoor': [11.200, 11.200, 96.800, 98.200, 99.200, 99.200, 99.400],
            'trojannn': [11.000, 11.400, 93.200, 94.600, 95.800, 96.400, 97.000],
            'imc': [11.200, 90.800, 96.800, 99.000, 99.000, 99.000, 99.000],
            'reflection_backdoor': [11.000, 11.200, 11.400, 11.400, 93.800, 95.400, 95.400],
            'targeted_backdoor': [11.200, 12.400, 33.400, 57.800, 85.400, 87.200, 88.200],
            'trojannet': [10.000, 12.600, 12.800, 10.200, 10.000, 10.000, 10.000],
            'bypass_embed': [10.600, 67.000, 78.400, 78.600, 86.400, 89.000, 90.000],
        },
    }
    x = np.linspace(2, 5, 4)
    for i, (key, value) in enumerate(attack_mapping.items()):
        x_list = np.array(x)
        y_list = np.array(y[args.dataset][key][1:len(x_list) + 1])
        fig.bar(x_list + (i - 4) * 0.1, y_list, color=color_dict[key], label=attack_mapping[key],
                width=0.1)
    # fig.set_legend()
    # fig.ax.get_legend().remove()
    fig.save(folder_path='./result/')
