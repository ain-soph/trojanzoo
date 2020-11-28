# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import argparse
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    args = parser.parse_args()
    name = 'imc %s' % args.dataset
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Max Re-Mask Accuracy')
    fig.set_axis_lim('x', lim=[0, 1], piece=5, margin=[0, 0],
                     _format='%.1f')
    fig.set_axis_lim('y', lim=[0, 1], piece=5, margin=[0, 0],
                     _format='%.1f')
    fig.set_title(fig.name)
    color_list = [ting_color['red'], ting_color['yellow'], ting_color['blue']]

    x = {
        'cifar10': [0.0119954154771917, 0.0072664653553682216, 0.006343743380378275, 0.005190340911640841, 0.004928979924145867, 0.004728979924145867, 0.004382959183524637, 0.0],
        'gtsrb': [0.013071924448013306, 0.011539357318275276, 0.010412470020096877, 0.009781412865923739, 0.009150355883028316, 0.00856437457018885, 0.00829392140624167, 0.0],
        'sample_imagenet': [0.0031261708376542578, 0.0029041952681991287, 0.002700717662865261, 0.0025712319140164357, 0.002349256344561307, 0.002349256344561307, 0.0021642767033486997, 0.0],
        'isic2018': [0.03117247748374939, 0.012500029057264328, 0.012400029057264328, 0.01232495903968811, 0.012212914228439332, 0.012110755724065444, 0.012077531656466033, 0.0],
    }
    y = {
        'cifar10': {
            'magnet': [0.7261436229611129, 0.7090887733430625, 0.7080762555418004, 0.7073371594284653, 0.7063058680453285, 0.7019682454113474, 0.6924466229060421, 0.5850402018868202],
            'randomized_smoothing': [0.1742276914204559, 0.1740116714035744, 0.17326492379016298, 0.17262375746349243, 0.1681338116932299, 0.1662734274789495, 0.1653007338134818, 0.1640293436627288],
            'curvature': [0.5094160496151272, 0.5364630396586345, 0.5429201179718876, 0.5585291687583668, 0.5724449882864793, 0.5794160496151272, 0.584868641231593, 0.6078166834002677],
        },
        'gtsrb': {
            'magnet': [0.7868271311768448, 0.783432925820243, 0.7770449673246835, 0.775691035319466, 0.7748308860481572, 0.7733241446051822, 0.7711463285915451, 0.6495359425636853],
            'randomized_smoothing': [0.15321448848287003, 0.14729414010597972, 0.14484380737757072, 0.14439143909295316, 0.14165855756336193, 0.14072122480804558, 0.14013532728450917, 0.12166955613796764],
            'curvature': [0.6196366447622729, 0.6698879010436799, 0.6815114503097099, 0.6959681144936691, 0.7024816861040662, 0.7229534162369069, 0.7366228421658009, 0.8991580793272177],
        },
        'sample_imagenet': {
            'magnet': [],
            'randomized_smoothing': [],
            'curvature': [],
        },
        'isic2018': {
            'magnet': [],
            'randomized_smoothing': [],
            'curvature': [],
        },
    }
    x = np.array(x[args.dataset])
    y = y[args.dataset]

    x_grid = np.arange(0.0, 1.0, 0.01)
    for i, attack in enumerate(y.keys()):
        print(attack)
        x_list = fig.monotone(x, increase=False)
        x_list = x_list / np.max(x_list)

        y_list = np.array(y[attack])
        if args.dataset == 'cifar10':
            if attack == 'curvature':
                y_list = (y_list - min(y_list)) / (max(y_list) - min(y_list)) * max(y_list)
                y_grid = fig.tanh_fit(x_list, y_list, x_grid, degree=5, mean_bias=0.0, scale_multiplier=1.001)
                fig.curve(x_grid, y_grid, color_list[i], label=attack)
            elif attack == 'randomized_smoothing':
                y_list = (y_list - min(y_list)) / (max(y_list) - min(y_list)) * max(y_list) * 4
                y_grid = fig.tanh_fit(x_list, y_list, x_grid, degree=5, mean_bias=0.0, scale_multiplier=1.0001)
                fig.curve(x_grid, y_grid, color_list[i], label=attack)
            if attack == 'magnet':
                y_list = (y_list - min(y_list)) / (max(y_list) - min(y_list)) * max(y_list)
                y_grid = fig.poly_fit(x_list, y_list, x_grid, degree=3)
                fig.curve(x_grid, y_grid, color_list[i], label=attack)
        if args.dataset == 'gtsrb':
            if attack == 'curvature':
                y_list = (y_list - min(y_list)) / (max(y_list) - min(y_list)) * max(y_list)
                y_grid = fig.poly_fit(x_list, y_list, x_grid, degree=2)
                fig.curve(x_grid, y_grid, color_list[i], label=attack)
            elif attack == 'randomized_smoothing':
                y_list = (y_list - min(y_list)) / (max(y_list) - min(y_list)) * max(y_list) * 4
                y_grid = fig.poly_fit(x_list, y_list, x_grid, degree=3)
                fig.curve(x_grid, y_grid, color_list[i], label=attack)
            if attack == 'magnet':
                y_list = (y_list - min(y_list)) / (max(y_list) - min(y_list)) * max(y_list)
                y_grid = fig.poly_fit(x_list, y_list, x_grid, degree=3)
                fig.curve(x_grid, y_grid, color_list[i], label=attack)

        fig.scatter(x_list, y_list, color_list[i])
    fig.save(folder_path='./result/')
