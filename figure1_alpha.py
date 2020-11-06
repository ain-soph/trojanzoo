# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    args = parser.parse_args()
    name = 'figure1 %s alpha' % args.dataset
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Transparency')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=10, margin=[0.05, 0.05],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'],
                  ting_color['green'], color['brown']['brown'], color['green']['army']]
    mark_list = ['H', '<', 'o', 'v', 's', 'p', '*', 'h', 'D']

    x = np.linspace(0.0, 1.0, 11)
    y = {
        'cifar10': {
            'badnet': [96.078, 96.078, 96.078, 96.078, 95.146, 94.118, 90.196, 83.810, 72.381, 52.577],
            'latent_backdoor': [100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 98.113],
            'trojannn': [100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 99.065, 97.196, 91.509, 62.617],
            'imc': [100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 100.000, 99.960, 99.220],
            'reflection_backdoor': [99.980, 99.810, 99.750, 99.430, 98.830, 97.330, 94.240, 87.400, 52.110, 10.660],
            'targeted_backdoor': [100.000, 100.000, 100.000, 100.000, 97.980, 95.146, 90.909, 79.412, 11.470, 10.680],
            # 'clean_label_pgd': [74.243, 51.167, 26.156, 13.037, 12.898, 12.712, 12.630, 12.661, 12.650, 10.540],
            'trojannet': [100, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352, 10.352],
            'bypassing': [95.320, 95.250, 94.370, 93.880, 93.300, 92.070, 90.460, 88.790, 74.320, 49.270],
        },
        'gtsrb': {
            'badnet': [95.469, 96.875, 95.312, 93.75, 93.75, 90.476, 88.525, 82.540, 65.634, 63.934],
            'latent_backdoor': [100, 100, 100, 100, 100, 100, 100, 100, 98.423, 91.803],
            'trojannn': [98.949, 98.761, 98.517, 98.086, 97.185, 95.777, 92.023, 79.223, 71.697, 51.952],
            'imc': [100, 100, 100, 100, 99.887, 99.662, 99.381, 98.461, 97.579, 88.645],
            'reflection_backdoor': [94.989, 90.709, 91.16, 83.54, 76.952, 67.98, 58.408, 50.282, 42.774, 3.979],
            'targeted_backdoor': [82.883, 78.866, 74.249, 61.374, 0.582, 0.582, 0.582, 0.582, 0.601, 0.601],
            # 'clean_label_pgd': [59.553, 42.962, 5.912, 2.196, 1.52, 1.989, 1.314, 0.845, 0.938, 0.976],
            'trojannet': [100, 1.014, 0.938, 0.582, 0.582, 0.582, 0.582, 0.582, 0.582, 0.582],
            'bypassing': [88.288, 87.819, 87.481, 86.768, 85.304, 85.39, 80.424, 75.713, 68.412, 49.831],
        },
        'sample_imagenet': {
            'badnet': [90.000, 90.000, 89.800, 88.200, 86.600, 81.400, 46.800, 11.600, 11.600, 11.600],
            'latent_backdoor': [97.400, 97.200, 96.800, 96.200, 96.400, 94.600, 93.200, 20.200, 11.400, 11.000],
            'trojannn': [95.200, 94.400, 93.200, 87.800, 11.800, 11.200, 11.200, 11.000, 11.200, 11.200],
            'imc': [98.400, 98.000, 96.800, 96.200, 95.800, 96.000, 95.000, 11.600, 11.400, 11.200],
            'reflection_backdoor': [94.600, 94.000, 11.400, 11.400, 11.400, 11.200, 11.200, 11.200, 11.000, 10.800],
            'targeted_backdoor': [82.800, 63.800, 33.400, 13.000, 11.800, 11.800, 11.800, 11.800, 11.800, 11.800],
            # 'clean_label_pgd': [59.553, 42.962, 5.912, 2.196, 1.52, 1.989, 1.314, 0.845, 0.938, 0.976],
            'trojannet': [100, 12.800, 12.800, 12.800, 12.600, 11.600, 11.400, 11.000, 11.000, 11.000],
            'bypassing': [82.600, 79.800, 78.400, 74.000, 72.400, 69.400, 46.800, 10.800, 10.600, 10.600],
        },
    }
    """Adjust plots for each dataset
    """
    if args.dataset == 'cifar10':
        for i, (key, value) in enumerate(y[args.dataset].items()):
            x_list = np.array(x[:len(value)])
            y_list = np.array(value)
            x_grid = np.linspace(0.0, 0.9, 5000)
            y_grid = np.linspace(0.0, 0.9, 5000)
            if key in ['imc', 'latent_backdoor', 'trojannn', 'reflection_backdoor', 'badnet', 'bypassing', 'clean_label_pgd', 'trojannet']:
                y_grid = fig.interp_fit(x_list, y_list, x_grid)
                if key in ['trojannet']:
                    y_grid += 5
                if key in ['clean_label_pgd']:
                    y_grid += 1
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=40)
            if key in ['targeted_backdoor']:
                x_grid = np.linspace(0.0, 0.9, 1000)
                y_grid = np.linspace(0.0, 0.9, 1000)
                y_grid_1 = fig.interp_fit(x_list, y_list, x_grid[:890])
                y_grid_2 = fig.poly_fit(x_list[8:], y_list[8:], x_grid[890:])
                y_grid = np.concatenate([y_grid_1, y_grid_2])
            y_grid[-1] = y_list[-1]

            fig.curve(x_grid, y_grid, color=color_list[i], label=key)
            fig.scatter(x_list, y_list, color=color_list[i], marker=mark_list[i])

    if args.dataset == 'sample_imagenet':
        for i, (key, value) in enumerate(y[args.dataset].items()):
            print(key)
            x_list = np.array(x[:len(value)])
            y_list = np.array(value)
            x_grid = np.linspace(0.0, 0.9, 900)
            y_grid = np.linspace(0.0, 0.9, 900)
            if key in ['badnet', 'targeted_backdoor']:
                x_grid = np.linspace(0.0, 0.9, 300)
                y_grid = np.linspace(0.0, 0.9, 300)
                y_grid = fig.interp_fit(x_list, y_list, x_grid)
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=40)
                if key in ['badnet', 'targeted_backdoor']:
                    y_grid += 1
            if key in ['bypassing']:
                x_grid = np.linspace(0.0, 0.9, 180)
                y_grid = np.linspace(0.0, 0.9, 180)
                y_grid = fig.poly_fit(x_list, y_list, x_grid, degree=5)
                y_grid[100:] = fig.interp_fit(x_list, y_list, x_grid[100:])
                y_grid[150:] = fig.poly_fit(x_list, y_list, x_grid[150:])
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=40)
                y_grid += 1
            if key in ['latent_backdoor']:
                x_grid = np.linspace(0.0, 0.9, 180)
                y_grid = np.linspace(0.0, 0.9, 180)
                y_grid[:140] = fig.interp_fit(x_list, y_list, x_grid[:140])
                y_grid[140:] = fig.exp_fit(x_list, y_list, x_grid[140:], degree=3)
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=10)
            if key in ['imc']:
                x_grid = np.linspace(0.0, 0.9, 180)
                y_grid = np.linspace(0.0, 0.9, 180)
                y_grid[:140] = fig.interp_fit(x_list, y_list, x_grid[:140])
                y_grid[140:] = fig.exp_fit(x_list, y_list, x_grid[140:], degree=1)
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=20)
            if key in ['trojannn']:
                x_grid = np.linspace(0.0, 0.9, 360)
                y_grid = np.linspace(0.0, 0.9, 360)
                y_grid[:160] = fig.interp_fit(x_list, y_list, x_grid[:160])
                y_grid[160:] = fig.exp_fit(x_list, y_list, x_grid[160:], degree=1)
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=40)
            if key in ['reflection_backdoor']:
                x_grid = np.linspace(0.0, 0.9, 1800)
                y_grid = np.linspace(0.0, 0.9, 1800)
                y_grid[:400] = fig.interp_fit(x_list, y_list, x_grid[:400])
                y_grid[400:] = fig.exp_fit(x_list, y_list, x_grid[400:], degree=3)
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=50)
            if key in ['trojannet']:
                x_grid = np.linspace(0.0, 0.9, 4500)
                y_grid = np.linspace(0.0, 0.9, 4500)
                y_grid = fig.exp_fit(x_list, y_list, x_grid, degree=4)
                y_grid = np.clip(y_grid, a_min=0.0, a_max=100.0)
                y_grid = fig.monotone(y_grid, increase=False)
                y_grid = fig.avg_smooth(y_grid, window=50)
                y_grid[0] = y_list[0]

            # if key not in ['latent_backdoor']: # check one line
            #     continue

            fig.curve(x_grid, y_grid, color=color_list[i])
            fig.scatter(x_list, y_list, color=color_list[i], marker=mark_list[i], label=key)
    fig.set_legend()
    fig.save('./result/')
