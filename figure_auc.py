# -*- coding: utf-8 -*-

from trojanzoo.plot import *

import numpy as np 
import pandas as pd
import warnings 
import os
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

def min_max_scaling(df, column_names=None):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    local_colnames = column_names if column_names is not None else df_nrom.columns
    for column in local_colnames:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm

def auc_graph(name, attack): 
    local_data = min_max_scaling(data[data["Model"] == attack], ["Difference", "Attack ACC"])
    # local_auc = auc(local_data["Attack ACC"], local_data["Difference"])
    # print(local_auc)

    fig = Figure(name=name)
    ax = fig.ax
    fig.set_axis_label('x', 'Efficacy')
    fig.set_axis_label('y', 'Specificity Loss')
    fig.set_axis_lim('x', lim=[0, 100.0], piece=5, margin=[0, 5],
                        _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5],
                        _format='%d')
    fig.set_title(fig.name)

    color_list = [ting_color['red_carrot'], ting_color['red_deep'], ting_color['yellow'],
                    ting_color['blue'], ting_color['blue_light'], ting_color['pink'],
                    ting_color['green'], color['brown']['brown'], color['green']['army']]
    mark_list = ['.', ',', 'o', 'v', 's', 'p', '*', 'h', 'D']
    x = np.array([i * 100 for i in local_data["Attack ACC"]])
    y = np.array([i * 100 for i in local_data["Difference"]])

    x_grid = np.linspace(0, 100, 1000)
    # y_grid = fig.poly_fit(x, y, x_grid, degree=2)
    # y_grid = fig.exp_fit(x, y, x_grid, increase=False)

    # x = 100 - np.array(x)
    x1 = np.linspace(0, 100, 100)
    y1 = x1

    i = 0

    fig.curve(x=x, y=y, color=color_list[i])
    fig.scatter(x=x, y=y, color=color_list[i], marker=mark_list[i])
    fig.curve(x=x1, y=y1, color=ting_color["grey"], linewidth=5, linestyle='--')

    fig.save("./")
    plt.show()

if __name__ == "__main__": 
    data = pd.read_excel("temp_data/auc_data_selected.xlsx")

    auc_graph("BadNet", "badnet")
    auc_graph("TrojanNN", "trojannn")
    auc_graph("Latent Backdoor", "latent_backdoor")
    auc_graph("IMC", "imc")