# -*- coding: utf-8 -*-

from trojanzoo import attack
from trojanzoo import model
from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    name = "Figure Model Architecture"
    fig = Figure(name)
    fig.set_axis_label('x', 'Attack')
    fig.set_axis_label('y', 'Attack Successful Rate')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['blue'], ting_color['yellow']]

    alpha_idx = 0
    # attack_list = ['badnet', 'latent', 'trojannn', 'imc', 'reflection', 'targeted', 'clean_label', 'trojannet', 'bypassing']

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
    model_dict = {
        'resnet18': 'ResNet18',
        'densenet121': 'DenseNet121',
        'vgg13': 'VGG13',
    }
    data = {
        'resnet18': {
            'badnet': [96.078, 96.078, 96.078, 96.078, 95.146, 94.118, 90.196, 83.810, 72.381, 52.577][alpha_idx],
            'latent_backdoor': [100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 98.113][alpha_idx],
            'trojannn': [100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 99.065, 97.196, 91.509, 62.617][alpha_idx],
            'imc': [100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 99.960, 99.220][alpha_idx],
            'reflection_backdoor': [99.980, 99.810, 99.750, 99.430, 98.830, 97.330, 94.240, 87.400, 52.110, 10.660][alpha_idx],
            'targeted_backdoor': [100.000, 100.000, 100.000, 100.000, 97.980, 95.146, 90.909, 79.412, 11.470, 10.680][alpha_idx],
            # 'trojannet': [74.243, 51.167, 26.156, 13.037, 12.898, 12.712, 12.630, 12.661, 12.650, 10.540][alpha_idx],
            'trojannet': [100, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352][alpha_idx],
            'bypass_embed': [95.320, 95.250, 94.370, 93.880, 93.300, 92.070, 90.460, 88.790, 74.320, 49.270][alpha_idx],
        },
        'densenet121': {
            'badnet': [90.82, 90.75, 89.76, 88.82, 86.75, 76.65, 68.66, 46.21, None, 10.56][alpha_idx],
            'latent_backdoor': [99.97, 99.97, 99.97, 99.97, 99.97, 99.73, 99.94, 99.83, 98.91, 10.65][alpha_idx],
            'trojannn': [98.85, 98.67, 98.43, 98.02, 96.51, 94.07, 67.26, 32.36, None, 10.56][alpha_idx],
            'imc': [100.0, 99.99, 100.0][alpha_idx],
            'reflection_backdoor': [98.460][alpha_idx],
            'targeted_backdoor': [94.190][alpha_idx],
            # 'trojannet': [14.280][alpha_idx],
            'trojannet': [100.000][alpha_idx],
            'bypass_embed': [93.30][alpha_idx],
        },
        'vgg13': {
            'badnet': [89.39, 87.92, 86.23, 83.89, 80.8, 72.89, 11.28, 11.39, 11.5, 10.47][alpha_idx],
            'latent_backdoor': [99.96, 99.99, 99.99, 99.95, 99.94, 99.79, 99.37, 97.44, 11.05, 10.41][alpha_idx],
            'trojannn': [89.54, 88.29, 86.19, 84.88, 81.49, 74.39, 36.67, 10.55, 11.06, 10.98][alpha_idx],
            'imc': [99.99, 99.99, 100.0, 99.93, 99.94, 99.79, 99.69, 99.27, None, None][alpha_idx],
            'reflection_backdoor': [87.860][alpha_idx],
            'targeted_backdoor': [93.590][alpha_idx],
            # 'trojannet': [0][alpha_idx],
            'trojannet': [100][alpha_idx],
            'bypass_embed': [89.280][alpha_idx],
        },
    }

    model_list = list(data.keys())

    attack_list = list(attack_mapping.keys())
    x = np.linspace(0, (len(attack_list) - 1) * 4, len(attack_list))
    for i, model_name in enumerate(model_list):
        y = np.array([data[model_name][key] for key in attack_list])
        fig.bar(x + i - 1.5, y, width=1, color=color_list[i], label=model_dict[model_name])

    fig.set_axis_lim('x', lim=[0, (len(attack_list) - 1) * 4], piece=len(attack_list) - 1, margin=[2.0, 2.0],
                     _format='%d')
    fig.set_axis_lim('y', lim=[80, 100], piece=5,
                     _format='%d')
    fig.ax.set_xticklabels([attack_mapping[key] for key in attack_list], rotation=0)
    # fig.ax.get_legend().remove()
    fig.set_legend(frameon=True)
    fig.save(folder_path='./result/')
    # plt.show()
