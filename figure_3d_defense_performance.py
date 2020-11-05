# -*- coding: utf-8 -*-

from trojanzoo import attack
from trojanzoo import model
from trojanzoo.plot import *

import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    name = 'Defense Performance'
    fig = Figure(name)
    # fig.set_axis_label('x', 'Defense')
    # fig.set_axis_label('y', 'Attack')
    # fig.set_axis_label('z', 'Defense Performance')

    # fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    x_defense = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    y_attack = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    z_performance = {
        x_defense[0]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[1]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[2]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[3]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[4]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[5]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[6]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[7]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        x_defense[8]: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    }

    x_mapping = {v: float(k) for k, v in enumerate(x_defense)}
    y_mapping = {v: float(k) for k, v in enumerate(y_attack)}

    x_list, y_list = np.meshgrid(x_defense, y_attack)
    x_list, y_list = x_list.ravel(), y_list.ravel()
    z_list = np.array([z_performance[x][int(y_mapping[y])] for x, y in zip(x_list, y_list)])

    x_pos = np.array([x_mapping[x] for x in x_list])
    y_pos = np.array([y_mapping[y] for y in y_list])
    bottom = np.zeros_like(z_list)

    width = depth = 0.5

    ax.bar3d(x_pos, y_pos, bottom, width, depth, z_list, shade=True)
    ax.set_xticklabels(x_defense, rotation=0)
    ax.set_yticklabels(y_attack, rotation=0)
    ax.set_title(name)

    plt.show()
