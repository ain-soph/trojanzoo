# -*- coding: utf-8 -*-

from package.plot.figure import Figure
from package.plot.colormap import ting_color
import numpy as np


fig = Figure('plot')
fig.set_axis_label('x', 'Watermark Size')
fig.set_axis_label('y', 'Max Re-Mask Accuracy')
fig.set_axis_lim('x', lim=[0.1, 0.8], piece=7, margin=[0.02, 0.02],
                 _format='%.1f')
fig.set_axis_lim('y', lim=[0, 1.0], piece=5,
                 _format='%.1f', margin=[0.0, 0.05])
fig.set_title(fig.name)

color_seq = ting_color


x = np.linspace(0.1, 0.8, 8)

y_base = np.array([0.99, 0.98, 1.0, 0.99, 1.0, 1.0, 0.99, 1.0])
y = np.array([0.99, 0.86, 0.70, 0.40, 0.33, 0.20, 0.13, 0.12])

fig.scatter(x, y, marker='s', linewidth=3, s=100,
            color=color_seq['red_deep'], zorder=3)
x_grid = np.linspace(0.1, 1, 100)
y_grid = [min(i, 1.0) for i in fig.tanh_fit(
    x, y, x_grid, degree=2, scale_multiplier=2.0)]
fig.curve(x_grid, y_grid, label='TrojanNN*', linewidth=3,
          color=color_seq['red_deep'], zorder=2)


fig.scatter(x, y_base, marker='o', linewidth=3, s=100,
            color=color_seq['red_carrot'], zorder=3)
x_grid = np.linspace(0, 1, 100)
y_grid = fig.tanh_fit(x, y_base, x_grid, degree=3)
fig.curve(x_grid, y_grid, label='TrojanNN', linewidth=3, linestyle='--',
          color=color_seq['red_carrot'], zorder=2)
fig.set_legend()
fig.save()
