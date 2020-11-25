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
    name = 'figure1 %s size confidence' % args.dataset
    fig = Figure(name, figsize=(5, 1))
    fig.set_axis_label('x', r'Trigger size ($|\mathit{m}|$)')
    fig.set_axis_label('y', 'Confidence')
    fig.set_axis_lim('x', lim=[2, 5], piece=3, margin=[0.5, 0.5],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=1, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title('')
    fig.ax.invert_yaxis()
    fig.ax.xaxis.tick_top()

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
            'badnet': [0.908, 0.949, 0.961, 0.957, 0.962, 0.965, 0.966],
            'latent_backdoor': [0.687, 0.995, 0.999, 1.000, 1.000, 1.000, 1.000],
            'trojannn': [0.865, 0.956, 0.967, 0.968, 0.974, 0.977, 0.986],
            'imc': [0.721, 0.950, 0.962, 0.964, 0.979, 0.985, 1.000],
            'reflection_backdoor': [0.883, 0.916, 0.953, 0.961, 0.976, 0.977, 0.987],
            'targeted_backdoor': [0.657, 0.674, 0.676, 0.681, 0.886, 0.918, 0.920],
            'trojannet': [0.0, 0.831, 0.843, 0.801, 0.804, 0.807, 0.804],
            'bypass_embed': [0.836, 0.906, 0.922, 0.924, 0.941, 0.944, 0.954],
        },
        'sample_imagenet': {
            'badnet': [0.532, 0.969, 0.985, 0.982, 0.983, 0.986, 0.989],
            'latent_backdoor': [0.487, 0.520, 0.996, 0.998, 0.999, 1.000, 1.000],
            'trojannn': [0.497, 0.544, 0.972, 0.984, 0.986, 0.987, 0.994],
            'imc': [0.647, 0.959, 0.983, 0.996, 0.997, 0.999, 1.000],
            'reflection_backdoor': [0.524, 0.540, 0.551, 0.551, 0.981, 0.982, 0.991],
            'targeted_backdoor': [0.657, 0.674, 0.676, 0.681, 0.886, 0.918, 0.920],
            'trojannet': [0.0, 0.347, 0.347, 0.347, 0.347, 0.348, 0.348],
            'bypass_embed': [0.438, 0.931, 0.958, 0.957, 0.977, 0.983, 0.986],
        },
    }
    x = np.linspace(2, 5, 4)
    for i, (key, value) in enumerate(attack_mapping.items()):
        x_list = np.array(x)
        y_list = np.array(y[args.dataset][key][1:len(x_list) + 1]) * 100
        fig.bar(x_list + (i - 4) * 0.1, y_list, color=color_dict[key], label=attack_mapping[key],
                width=0.1)
    # fig.set_legend()
    # fig.ax.get_legend().remove()
    fig.save(folder_path='./result/')
