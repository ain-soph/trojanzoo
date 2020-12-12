# -*- coding: utf-8 -*-

from trojanzoo.plot import *
from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.defense import Defense_Backdoor

import numpy as np
from typing import Dict

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack(), Parser_Defense())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    defense: Defense_Backdoor = parser.module_list['defense']

    file_name = f'{defense.folder_path}{defense.get_filename()}.npy'
    _dict: Dict[str, np.ndarray] = np.load(file_name, allow_pickle=True).item()

    # ------------------------------------------------------------------------ #

    fig = Figure(name=f'dist {defense.name} {attack.name}')
    fig.set_axis_label('x', 'Entropy')
    fig.set_axis_label('y', 'PDF')
    fig.set_axis_lim('x', lim=[0, 1.0], piece=5, margin=[0.0, 0.05],
                     _format='%.1f')
    y_lim = 20.0
    if attack.name == 'badnet':
        y_lim = 5.0
    fig.set_axis_lim('y', lim=[0, y_lim], piece=5, margin=[0.0, 0.0],
                     _format='%.1f')

    color_list = [ting_color['red_carrot'], ting_color['blue']]
    fig.set_title()

    x_grid = np.linspace(0.0, 1.0, 1000)
    bin_grid = np.linspace(0.0, 1.0, 20)

    fig.hist(_dict['clean'], bins=bin_grid, color=ting_color['red_carrot'], alpha=0.3)
    fig.hist(_dict['poison'], bins=bin_grid, color=ting_color['blue'], alpha=0.3)

    y_grid = fig.gaussian_kde(_dict['clean'], x_grid)
    fig.curve(x_grid, y_grid, color=ting_color['red_carrot'], label='clean')

    y_grid = fig.gaussian_kde(_dict['poison'], x_grid)
    fig.curve(x_grid, y_grid, color=ting_color['blue'], label='poison')

    fig.save('./result/dist/')
