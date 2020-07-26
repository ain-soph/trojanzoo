# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    fig = Figure('figure1_alpha')
    fig.set_axis_label('x', 'Trigger Transparency')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=10, margin=[0.02, 0],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 100], piece=5,
                     _format='%d', margin=[0.0, 0.0])
    fig.set_title(fig.name)

    color_list = [ting_color['red'], ting_color['yellow'], ting_color['green']]

    x = np.linspace(0.0, 0.9, 10)
    y = {
        'badnet': [78.710, 76.910, 73.940, 67.310, 61.980, 54.450, 45.330, 37.290, 29.610, 22.960],
        'latent_backdoor': [97.140, 95.450, 90.080, 86.490, 82.010, 71.660, 55.440, 23.150, 10.190, 10.280],
        'trojannn': [73.170, 71.290, 69.170, 66.620, 62.420, 53.850, 46.190, 37.630, 30.360, 12.290]
    }

    for i, (key, value) in enumerate(y.items()):
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i])
    fig.save('./result/')
