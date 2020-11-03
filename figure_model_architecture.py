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

    color_list = [ting_color['red_carrot'], ting_color['blue'], ting_color['green']]

    


    # attack_list = ['badnet', 'latent', 'trojannn', 'imc', 'reflection', 'targeted', 'clean_label', 'trojannet', 'bypassing']
    attack_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    data = {
        'resnet18': {
            attack_list[0]: 96.078,
            attack_list[1]: 100.000,
            attack_list[2]: 100.000,
            attack_list[3]: 100.000,
            attack_list[4]: 67.3,
            attack_list[5]: 67.3,
            attack_list[6]: 67.3,
            attack_list[7]: 67.3,
            attack_list[8]: 67.3,
        },
        'vgg13': {
            attack_list[0]: 89.390,
            attack_list[1]: 99.960,
            attack_list[2]: 89.540,
            attack_list[3]: 99.990,
            attack_list[4]: 67.3,
            attack_list[5]: 67.3,
            attack_list[6]: 67.3,
            attack_list[7]: 67.3,
            attack_list[8]: 67.3,
        },
        'densenet121': {
            attack_list[0]: 90.820,
            attack_list[1]: 99.970,
            attack_list[2]: 98.850,
            attack_list[3]: 100.000,
            attack_list[4]: 67.3,
            attack_list[5]: 67.3,
            attack_list[6]: 67.3,
            attack_list[7]: 67.3,
            attack_list[8]: 67.3,
        },
    }

    model_list = list(data.keys())

    x = np.linspace(0, (len(attack_list) - 1) * 4, len(attack_list))
    for i, model_name in enumerate(model_list):
        y = np.array(list(data[model_name].values()))
        fig.bar(x + i - 1.5, y, width=1, color=color_list[i], label=model_name)

    fig.set_axis_lim('x', lim=[0, (len(attack_list) - 1) * 4], piece=len(attack_list) - 1, margin=[2.0, 2.0],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5,
                     _format='%d')
    fig.ax.set_xticklabels(attack_list, rotation=0)
    fig.set_legend()
    fig.save('./result/')
