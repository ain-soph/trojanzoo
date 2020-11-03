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
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    if args.dataset == 'gtsrb':
        fig.set_axis_lim('x', lim=[0, 10], piece=10, margin=[0, 0.5],
                         _format='%d')
    else:
        fig.set_axis_lim('x', lim=[0, 7], piece=7, margin=[0, 0.5],
                         _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'],
                  ting_color['green'], color['brown']['brown'], color['green']['army']]
    mark_list = ['H', ',', 'o', 'v', 's', 'p', '*', 'h', 'D']

    x = np.linspace(1, 10, 10)
    y = {
        'cifar10': {
            'badnet': [61.520, 70.520, 72.381, 77.350, 79.380, 81.040, 81.560],
            'latent_backdoor': [10.720, 99.250, 100.000, 100.000, 100.000, 100.000, 100.000],
            'trojannn': [46.600, 87.770, 91.509, 91.910, 92.360, 93.520, 94.990],
            'imc': [58.550, 99.660, 99.960, 99.990, 100.000, 100.000, 100.000],
            'reflection_backdoor': [44.560, 64.300, 79.240, 88.150, 92.920, 94.000, 96.390],
            'targeted_backdoor': [10.940, 11.140, 11.470, 11.760, 33.290, 44.450, 49.000],
            'clean_label_pgd': [12.190, 12.410, 12.650, 13.040, 13.240, 13.030, 14.650],
            'trojannet': [10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352],
            'bypassing': [66.700, 74.270, 74.320, 78.520, 83.340, 83.650, 85.610],
        },
        'gtsrb': {
            'badnet': [0.619, 61.543, 65.634, 71.415, 71.772, 71.753, 72.954, 71.565, 73.949, 75],
            'latent_backdoor': [99.625, 99.23, 98.423, 99.249, 99.662, 99.887, 99.925, 99.887, 99.925, 99.962],
            'trojannn': [0.601, 57.508, 71.697, 69.67, 72.11, 73.011, 78.96, 81.963, 82.658, 83.483],
            'imc': [21.34, 92.399, 97.579, 95.89, 96.509, 98.986, 99.095, 98.874, 98.911, 98.968],
            'reflection_backdoor': [3.003, 38.589, 42.774, 48.311, 53.848, 62.218, 64.492, 74.437, 72.879, 85.511],
            'targeted_backdoor': [0.619, 0.619, 0.601, 0.619, 0.638, 0.601, 0.601, 0.788, 0.807, 0.77],
            'clean_label_pgd': [1.858, 1.464, 0.938, 1.745, 0.601, 1.014, 0.582, 1.839, 1.276, 0.807],
            'trojannet': [0.582, 0.582, 0.582, 0.582, 0.582, 0.582, 0.582, 0.563],
            'bypassing': [7.432, 61.974, 68.412, 73.78, 73.142, 73.104, 74.474, 76.52, 79.279, 78.829],
        },
        'sample_imagenet': {
            'badnet': [11.400, 83.400, 89.800, 91.200, 91.400, 91.400, 91.400],
            'latent_backdoor': [11.200, 11.200, 96.800, 98.200, 99.200, 99.200, 99.400],
            'trojannn': [11.000, 11.400, 93.200, 94.600, 95.800, 96.400, 97.000],
            'imc': [11.200, 90.800, 96.800, 99.000, 99.000, 99.000, 99.000],
            'reflection_backdoor': [11.000, 11.200, 11.400, 11.400, 93.800, 95.400, 95.400],
            'targeted_backdoor': [11.200, 12.400, 33.400, 57.800, 85.400, 87.200, 88.200],
            'trojannet': [10.000, 12.600, 12.800, 10.200, 10.000, 10.000, 10.000],
            'bypassing': [10.600, 67.000, 78.400, 78.600, 86.400, 89.000, 90.000],
        },
    }
    for i, (key, value) in enumerate(y[args.dataset].items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i], marker=mark_list[i])
    fig.save('./result/')
