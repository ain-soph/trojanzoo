# -*- coding: utf-8 -*-


from trojanzoo.plot import *
from trojanzoo.plot.font import *

import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    data = {
        'ASR': {
            'original': [83.97, 19.79],
            'adaptive': [99.840, 98.630],
        },
        'Clean Acc': {
            'original': [84.57, 84.39],
            'adaptive': [75.940, 82.970],
        }
    }
    _fig, _axarr = plt.subplots(2, 1, figsize=(5, 2.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    color_list = [ting_color['red_carrot'], ting_color['blue_light']]
    size_list = np.array([3, 6])
    width = 0.8
    for i, mode in enumerate(list(data.keys())):
        fig = Figure(name='adaptive-magnet', fig=_fig, ax=_axarr[i])
        fig.set_title('')
        if mode == 'Clean Acc':
            fig.set_axis_label('x', r'Trigger size ($|\mathit{m}|$)')
            fig.set_axis_label('y', 'Clean Acc (%)')
            fig.set_axis_lim('x', lim=[3, 6], margin=[1.5, 1.5], piece=1, _format='%d')
            fig.set_axis_lim('y', lim=[0, 100], margin=[0, 0], piece=1, _format='%d')
            _axarr[i].invert_yaxis()
            _axarr[i].xaxis.tick_top()
        else:
            # fig.set_axis_label('x', r'Trigger size ($|\mathit{m}|$)')
            fig.set_axis_label('y', 'ASR (%)')
            # fig.set_axis_lim('x', lim=[3, 6], margin=[2, 3], piece=1, _format='%d')
            fig.set_axis_lim('y', lim=[0, 100], margin=[0, 0], piece=5, _format='%d')
        for j, label in enumerate(list(data[mode].keys())):
            y = np.array(data[mode][label])
            if mode == 'Clean Acc':
                fig.bar(size_list + (j - 1) * width, y, label=label, width=width,
                        color='white', edgecolor=color_list[j], linewidth=2)
            else:
                fig.bar(size_list + (j - 1) * width, y, label=label, width=width, color=color_list[j])
    # fig.set_legend(prop=palatino_bold, loc='upper right')

    # fig.ax.set_xticklabels([r'$3\times 3$', r'$6\times 6$'], rotation=0)
    fig.ax.set_xticklabels([r'', r''], rotation=0)

    fig.save(folder_path='./result/adaptive/')
