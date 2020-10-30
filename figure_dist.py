# -*- coding: utf-8 -*-

from re import sub
from trojanzoo.plot import *

import numpy as np
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    _dict = {}
    _dict['neural_cleanse'] = {}

    _dict['neural_cleanse']['loss'] = [
        [0.0006, 0.0053, 0.0050, 0.0062, 0.0075, 0.0052, 0.0054, 0.0051, 0.0060, 0.0054],
        [0.0040, 0.0008, 0.0040, 0.0058, 0.0054, 0.0057, 0.0053, 0.0057, 0.0063, 0.0063],
        [0.0037, 0.0048, 0.0008, 0.0055, 0.0077, 0.0055, 0.0048, 0.0067, 0.0062, 0.0061],
        [0.0047, 0.0049, 0.0044, 0.0008, 0.0075, 0.0052, 0.0053, 0.0067, 0.0062, 0.0059],
        [0.0037, 0.0051, 0.0056, 0.0055, 0.0007, 0.0054, 0.0048, 0.0053, 0.0053, 0.0056],
        [0.0044, 0.0047, 0.0050, 0.0064, 0.0076, 0.0010, 0.0053, 0.0055, 0.0063, 0.0058],
        [0.0049, 0.0045, 0.0052, 0.0057, 0.0059, 0.0055, 0.0008, 0.0059, 0.0066, 0.0063],
        [0.0038, 0.0045, 0.0045, 0.0063, 0.0076, 0.0052, 0.0054, 0.0007, 0.0066, 0.0060],
        [0.0039, 0.0048, 0.0045, 0.0058, 0.0082, 0.0053, 0.0055, 0.0054, 0.0007, 0.0062],
        [0.0037, 0.0053, 0.0044, 0.0061, 0.0074, 0.0056, 0.0055, 0.0058, 0.0065, 0.0009],
    ]
    _dict['neural_cleanse']['mask_norm'] = [
        [5.2482, 40.1884, 28.8140, 44.6914, 40.2488, 30.8517, 47.2106, 46.3075, 49.8708, 46.6875],
        [31.8815, 6.4037, 33.6257, 46.6820, 39.3330, 32.8310, 54.4405, 46.4379, 42.7248, 50.4681],
        [30.3199, 46.0209, 5.8217, 44.8451, 45.0154, 31.9845, 50.6955, 50.2948, 48.5339, 47.5366],
        [32.6503, 39.1451, 31.5808, 5.6404, 46.2550, 30.7051, 51.5417, 47.2029, 44.8375, 48.5399],
        [28.0862, 41.6171, 30.8379, 42.8326, 6.1434, 30.8143, 49.4135, 48.3746, 47.2351, 47.2109],
        [31.8253, 43.8442, 31.9409, 44.5664, 44.9137, 6.2544, 52.6821, 47.7908, 52.3733, 52.8196],
        [30.6184, 40.3103, 32.1240, 45.0711, 34.4575, 31.8603, 6.9159, 49.4014, 47.4417, 45.9846],
        [30.5478, 44.7078, 29.5927, 42.2625, 37.9957, 29.6467, 47.4091, 4.8385, 47.8706, 43.5646],
        [29.4252, 44.9401, 34.6991, 43.1373, 38.5358, 32.7187, 53.5929, 47.9920, 5.2826, 46.1599],
        [30.7670, 40.1997, 32.2215, 45.5490, 43.5742, 34.8875, 51.7282, 48.2348, 44.3677, 5.1852],
    ]

    for defense in _dict.keys():
        for metric in ['loss', 'mask_norm']:
            fig = Figure(name=f'{defense} {metric}')
            fig.set_axis_label('x', metric)
            fig.set_axis_label('y', 'PDF')
            x_grid = np.linspace(0, 1e-2, 1000)
            bin_grid = np.linspace(0, 1e-2, 20)
            if metric == 'loss':
                fig.set_axis_lim('x', lim=[0, 10], piece=5, margin=[0.0, 0.05],
                                 _format='%.3f')
                fig.set_axis_lim('y', lim=[0, 4.0], piece=5, margin=[0.0, 0.2],
                                 _format='%.1f')
                x_grid = np.linspace(0, 1e-2, 1000)
                bin_grid = np.linspace(0, 1e-2, 20)
            elif metric == 'mask_norm':
                fig.set_axis_lim('x', lim=[0, 60], piece=5, margin=[0.0, 0.05],
                                 _format='%.1f')
                fig.set_axis_lim('y', lim=[0, 0.5], piece=5, margin=[0.0, 0.025],
                                 _format='%.1f')
                x_grid = np.linspace(0, 60, 1000)
                bin_grid = np.linspace(0, 60, 20)
            fig.set_title()
            color_list = [ting_color['red_carrot'], ting_color['blue']]

            data = _dict[defense][metric]
            clean = []
            poison = []
            for i, sub_list in enumerate(data):
                for j, element in enumerate(sub_list):
                    if i == j:
                        poison.append(element)
                    else:
                        clean.append(element)

            clean = np.array(clean)
            poison = np.array(poison)
            if metric == 'loss':
                clean *= 1000
                poison *= 1000
                x_grid *= 1000
                bin_grid *= 1000

            fig.hist(clean, bins=bin_grid, color=ting_color['red_carrot'], alpha=0.3)
            fig.hist(poison, bins=bin_grid, color=ting_color['blue'], alpha=0.3)

            if metric == 'mask_norm':
                rv = stats.norm(loc=np.mean(clean), scale=np.std(clean))
                y_grid = rv.pdf(x_grid)
            else:
                y_grid = fig.gaussian_kde(clean, x_grid)
            fig.curve(x_grid, y_grid, color=ting_color['red_carrot'], label='clean')

            y_grid = fig.gaussian_kde(poison, x_grid)
            fig.curve(x_grid, y_grid, color=ting_color['blue'], label='poison')

            fig.save("./result/dist/")
