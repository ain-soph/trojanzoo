# -*- coding: utf-8 -*-, from trojanzoo.plot import *

from trojanzoo.plot import *

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    parser.add_argument('-c', '--confidence', dest='confidence', action='store_true')
    args = parser.parse_args()
    name = 'figure1 %s size' % args.dataset
    if args.confidence:
        name += ' confidence'
    fig = Figure(name)
    fig.set_axis_label('x', 'Trigger Size')
    fig.set_axis_label('y', 'Misclassification Confidence' if args.confidence else 'Max Re-Mask Accuracy')
    if args.dataset == 'cifar10':
        fig.set_axis_lim('x', lim=[0, 7], piece=7, margin=[0, 0.5],
                         _format='%d')
    if args.dataset == 'gtsrb':
        fig.set_axis_lim('x', lim=[0, 10], piece=10, margin=[0, 0.5],
                         _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                  ting_color['blue'], ting_color['blue_light'], ting_color['pink'],
                  ting_color['green'], color['brown']['brown'], color['green']['army']]
    mark_list = ['.', ',', 'o', 'v', 's', 'p', '*', 'h', 'D']

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
        # 'gtsrb': {
        #     'badnet': [42.68, 56.757, 53.053, 59.791, 56.044, 55.03, 61.787, 73.724, 76.107, 79.936],
        #     'latent_backdoor': [9.572, 92.080, 99.981, 99.944, 100, 100, 100, 100, 100, 100],
        #     'trojannn': [37.125, 59.816, 57.789, 58.146, 73.048, 94.125, 98.104, 84.441, 93.919, 95.420],
        #     'targeted_backdoor': [23.161, 40.165, 40.897, 41.498, 40.034, 46.021, 50.544, 55.011, 61.787, 60.773],
        # },
    }
    z = {
        # 'cifar10': {
        #     'badnet': [0.652851402759552, 0.5328619480133057, 0.6471461653709412, 0.7925588488578796, 0.8418892621994019, 0.7844262719154358, 0.8106977343559265],
        #     'latent_backdoor': [0.5402058362960815, 0.9261912703514099, 0.9638524651527405, 0.9476844668388367, 0.9828152060508728, 0.9868102073669434, 0.8296664357185364],
        #     'trojannn': [0.4948200583457947, 0.5762385725975037, 0.7242116332054138, 0.6653879880905151, 0.580232560634613, 0.508546769618988, 0.5213910937309265],
        #     'imc': [0.5129220485687256, 0.7297095060348511, 0.6254073977470398, 0.4126419723033905],
        #     'clean_label_pgd': [0.8591371178627014, 0.8928311467170715, 0.9218209385871887, 0.931251049041748, 0.9373565912246704, 0.9461389780044556, 0.9672442078590393],
        # },
    }
    for key in z.keys():
        for sub_key in z[key].keys():
            for i in range(len(z[key][sub_key])):
                z[key][sub_key][i] *= 100
    y = z if args.confidence else y
    for i, (key, value) in enumerate(y[args.dataset].items()):
        print(key)
        fig.curve(x[:len(value)], value, color=color_list[i], label=key)
        fig.scatter(x[:len(value)], value, color=color_list[i], marker=mark_list[i])
    fig.save('./result/')
