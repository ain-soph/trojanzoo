#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .fonts import *
from trojanzoo.utils import to_numpy

import os
import numpy as np
import torch

import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.container import BarContainer
import seaborn
import scipy.stats as stats

from scipy.interpolate import UnivariateSpline
# from scipy.optimize import curve_fit

rc('font', family='serif', serif='Palatino', weight='bold')
rc('svg', image_inline=True, fonttype='none')
rc('pdf', fonttype=42)
rc('ps', fonttype=42)
rc('mathtext', fontset='cm')


class Figure:
    def __init__(self, name: str, folder_path: str = None, fig: Figure = None, ax: Axes = None, figsize: tuple[float, float] = (5, 2.5), tex=False):
        super(Figure, self).__init__()
        if tex:
            rc('text', usetex=True)
        self.name: str = name
        self.folder_path: str = folder_path
        if folder_path is None:
            self.folder_path = './output/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.fig: Figure = fig
        self.ax: Axes = ax
        if fig is None and ax is None:
            self.fig: Figure = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(True)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(axis='y', linewidth=1)
        self.ax.set_axisbelow(True)

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])

    def set_legend(self, *args, frameon: bool = True, edgecolor='white', framealpha=1.0,
                   fontsize=11, fontstyle='italic', **kwargs) -> None:
        self.ax.legend(*args, frameon=frameon, edgecolor=edgecolor, framealpha=framealpha, **kwargs)
        plt.setp(self.ax.get_legend().get_texts(), fontsize=fontsize, fontstyle=fontstyle)

    def set_axis_label(self, axis: str, text: str, fontsize: int = 12,
                       family='serif', font='Palatino', weight='bold', **kwargs):
        getattr(self.ax, f'set_{axis}label')(text, fontsize=fontsize,
                                             family=family, font=font, weight=weight, **kwargs)

    def set_title(self, text: str = None, fontsize: int = 16) -> None:
        if text is None:
            text = self.name
        self.ax.set_title(text, fontsize=fontsize)

    def save(self, path: str = None, folder_path: str = None, ext: str = 'pdf') -> None:
        if path is None:
            if folder_path is None:
                folder_path = self.folder_path
            path = f'{folder_path}{self.name}.{ext}'
        else:
            folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.fig.savefig(path, dpi=100, bbox_inches='tight')

    def set_axis_lim(self, axis: str, lim: list[float] = [0.0, 1.0], margin: list[float] = [0.0, 0.0],
                     piece: int = 10, _format: str = '%.1f',
                     fontsize: int = 11) -> None:
        if _format == 'integer':
            _format = '%d'
        lim_func = getattr(self.ax, f'set_{axis}lim')
        set_ticks_func = getattr(self.ax, f'set_{axis}ticks')

        def format_func(_str):
            getattr(self.ax, f'{axis}axis').set_major_formatter(
                ticker.FormatStrFormatter(_str))
        ticks = np.append(
            np.arange(lim[0], lim[1], (lim[1] - lim[0]) / piece), lim[1])
        final_lim = [lim[0] - margin[0], lim[1] + margin[1]]
        lim_func(final_lim)
        set_ticks_func(ticks)
        ticks = getattr(self.ax, f'get_{axis}ticks')()
        set_ticklabels_func = getattr(self.ax, f'set_{axis}ticklabels')
        set_ticklabels_func(ticks, fontsize=fontsize)
        format_func(_format)

    def curve(self, x: np.ndarray, y: np.ndarray, color: str = 'black', linewidth: int = 2,
              label: str = None, markerfacecolor: str = 'white', linestyle: str = '-', zorder: int = 1, **kwargs) -> Line2D:
        # linestyle marker markeredgecolor markeredgewidth markerfacecolor markersize alpha
        ax = seaborn.lineplot(x, y, ax=self.ax, color=color, linewidth=linewidth,
                              markerfacecolor=markerfacecolor, zorder=zorder, **kwargs)
        line: Line2D = ax.get_lines()[-1]
        line.set_linestyle(linestyle)
        if label is not None:
            self.curve_legend(label=label, color=color, linewidth=linewidth, **kwargs)
        return line

    def curve_legend(self, label: str = None, color: str = 'black', linewidth: int = 2, markerfacecolor: str = 'white', **kwargs) -> Line2D:
        # linestyle marker markeredgecolor markeredgewidth markerfacecolor markersize alpha
        line, = self.ax.plot([], [], color=color, linewidth=linewidth, markeredgewidth=linewidth, markeredgecolor=color,
                             label=label, markerfacecolor=markerfacecolor, **kwargs)
        return line

    def scatter(self, x: np.ndarray, y: np.ndarray, color: str = 'black', linewidth: int = 2,
                label: str = None, marker: str = 'D', facecolor: str = 'white', zorder: int = 3, **kwargs):
        # marker markeredgecolor markeredgewidth markerfacecolor markersize alpha
        if label is not None:
            self.curve_legend(label=label, color=color, linewidth=linewidth, marker=marker, **kwargs)
        return self.ax.scatter(x, y, color=color, linewidth=linewidth, marker=marker, facecolor=facecolor, zorder=zorder, **kwargs)

    def add_subplot(self, projection=None):
        if projection is not None:
            return self.fig.add_subplot(projection=projection)

# Markers
# '.' point marker
# ',' pixel marker
# 'o' circle marker
# 'v' triangle_down marker
# '^' triangle_up marker
# '<' triangle_left marker
# '>' triangle_right marker
# '1' tri_down marker
# '2' tri_up marker
# '3' tri_left marker
# '4' tri_right marker
# 's' square marker
# 'p' pentagon marker
# '*' star marker
# 'h' hexagon1 marker
# 'H' hexagon2 marker
# '+' plus marker
# 'x' x marker
# 'D' diamond marker
# 'd' thin_diamond marker
# '|' vline marker
# '_' hline marker

# Line Styles
# '-'     solid line style
# '--'    dashed line style
# '-.'    dash-dot line style
# ':'     dotted line style

    def bar(self, x: np.ndarray, y: np.ndarray, color: str = 'black', width: float = 0.2,
            align: str = 'edge', edgecolor: str = 'white', label: str = None, **kwargs) -> BarContainer:
        # facecolor edgewidth alpha
        return self.ax.bar(x, y, color=color, width=width, align=align, edgecolor=edgecolor, label=label, **kwargs)

    def bar3d(self, x: np.ndarray, y: np.ndarray, z: np.array, color: str = 'black', size: tuple[float, float] = 0.5,
              label: str = None, **kwargs) -> BarContainer:
        # facecolor edgewidth alpha
        if isinstance(size, (float, int)):
            size = [size, size]
        return self.ax.bar3d(x=x, y=y, z=np.zeros_like(x),
                             dx=np.ones_like(x) * size[0], dy=np.ones_like(y) * size[1], dz=z,
                             color=color, label=label, **kwargs)

    def hist(self, x: np.ndarray, bins: list[float] = None, normed: bool = True, **kwargs):
        return self.ax.hist(x, bins=bins, normed=normed, **kwargs)

    def autolabel(self, rects: BarContainer, above: bool = True,
                  fontsize: int = 6):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = int(rect.get_height())
            offset = 3 if above else -13
            self.ax.annotate('%d' % (abs(height)),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, offset),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=fontsize)

    @staticmethod
    def get_roc_curve(label, pred) -> tuple[list[float], list[float]]:

        total_inst = len(label)
        total_pos_inst = len(np.where(label == 1)[0])

        assert len(label) == len(pred)
        # true positive rates and false positive rates
        tprs, fprs, thresholds = [], [], []

        # iterate over all positive thresholds
        for threshold in np.unique(pred):

            pred_pos_idx = np.where(pred >= threshold)[0]

            # number of predicted positive instances
            pred_pos_inst = len(pred_pos_idx)
            # number of true positive instances
            true_pos_inst = np.count_nonzero(label[pred_pos_idx])

            tpr = true_pos_inst * 1. / total_pos_inst * 1.
            fpr = (pred_pos_inst - true_pos_inst) * \
                1. / (total_inst - total_pos_inst) * 1.
            tprs.append(tpr)
            fprs.append(fpr)
            thresholds.append(threshold)

        return fprs, tprs

    @staticmethod
    def sort(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(x)
        return np.array(x)[idx], np.array(y)[idx]

    @staticmethod
    def normalize(x: np.ndarray, _min: float = None, _max: float = None, tgt_min: float = 0.0, tgt_max: float = 1.0) -> np.ndarray:
        x = to_numpy(x)
        if _min is None:
            _min = x.min()
        if _max is None:
            _max = x.max()
        x = (x - _min) / (_max - _min) * (tgt_max - tgt_min) + tgt_min
        return x

    @staticmethod
    def groups_err_bar(x: np.ndarray, y: np.ndarray) -> dict[float, np.ndarray]:
        y_dict = {}
        for _x in set(x):
            y_dict[_x] = np.array([y[t] for t in range(len(y)) if x[t] == _x])
        return y_dict

    @staticmethod
    def flatten_err_bar(y_dict: dict[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = []
        y = []
        for _x in y_dict.keys():
            for _y in y_dict[_x]:
                x.append(_x)
                y.append(_y)
        return np.array(x), np.array(y)

    @classmethod
    def normalize_err_bar(cls, x: np.ndarray, y: np.ndarray):
        x = cls.normalize(x)
        y_dict = cls.groups_err_bar(x, y)
        y_mean = np.array([y_dict[_x].mean()
                           for _x in np.sort(list(y_dict.keys()))])
        y_norm = cls.normalize(y_mean)
        y_dict = cls.adjust_err_bar(y_dict, y_norm - y_mean)
        return cls.flatten_err_bar(y_dict)

    @classmethod
    def avg_smooth_err_bar(cls, x: np.ndarray, y: np.ndarray, window: int = 3):
        y_dict = cls.groups_err_bar(x, y)
        y_mean = np.array([y_dict[_x].mean()
                           for _x in np.sort(list(y_dict.keys()))])
        y_smooth = cls.avg_smooth(y_mean, window=window)
        y_dict = cls.adjust_err_bar(y_dict, y_smooth - y_mean)
        return cls.flatten_err_bar(y_dict)

    @staticmethod
    def adjust_err_bar(y_dict: dict[float, np.ndarray], mean: np.ndarray = None, std: np.ndarray = None) -> dict[float, np.ndarray]:
        sort_keys = np.sort(list(y_dict.keys()))
        if isinstance(mean, float):
            mean = mean * np.ones(len(sort_keys))
        if isinstance(std, float):
            std = std * np.ones(len(sort_keys))
        for i in range(len(sort_keys)):
            key = sort_keys[i]
            if mean:
                y_dict[key] = y_dict[key] + mean[i]
            if std:
                y_dict[key] = y_dict[key].mean() + \
                    (y_dict[key] - y_dict[key].mean()) * std[i]
        return y_dict

    @staticmethod
    def avg_smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
        _x = torch.as_tensor(x)
        new_x = torch.zeros_like(_x)
        for i in range(len(_x)):
            if i < window // 2:
                new_x[i] = (_x[0] * (window // 2 - i)
                            + _x[: i + (window + 1) // 2].sum()) / window
            elif i >= len(_x) - (window - 1) // 2:
                new_x[i] = (_x[-1] * ((window + 1) // 2 - len(_x) + i) +
                            _x[i - window // 2:].sum()) / window
            else:
                new_x[i] = _x[i - window // 2:i + 1 + (window - 1) // 2].mean()
        return to_numpy(new_x) if isinstance(x, np.ndarray) else new_x

    @staticmethod
    def poly_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, degree: int = 1) -> np.ndarray:
        fit_data = to_numpy(y)
        z = np.polyfit(x, fit_data, degree)
        y_grid = np.polyval(z, x_grid)
        return y_grid

    @staticmethod
    def tanh_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
                 degree: int = 1, mean_bias: float = 0.0, scale_multiplier: float = 1.0) -> np.ndarray:
        mean = (max(y) + min(y)) / 2 + mean_bias
        scale = max(abs(y - mean)) * scale_multiplier
        fit_data = to_numpy(torch.as_tensor((y - mean) / scale).atanh())
        z = np.polyfit(x, fit_data, degree)
        y_grid = np.tanh(np.polyval(z, x_grid)) * scale + mean
        return y_grid

    @staticmethod
    def atan_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
                 degree: int = 1, mean_bias: float = 0.0, scale_multiplier: float = 1.0) -> np.ndarray:
        mean = (max(y) + min(y)) / 2 + mean_bias
        scale = max(abs(y - mean)) * scale_multiplier
        fit_data = to_numpy(torch.as_tensor((y - mean) / scale).tan())
        z = np.polyfit(x, fit_data, degree)
        y_grid = np.tanh(np.polyval(z, x_grid)) * scale + mean
        return y_grid

    @staticmethod
    def exp_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
                degree: int = 1, increase: bool = True, epsilon: float = 0.01) -> np.ndarray:
        y_max = max(y)
        y_min = min(y)
        if increase:
            fit_data = np.log(y + epsilon - y_min)
        else:
            fit_data = np.log(y_max + epsilon - y)

        z = np.polyfit(x, fit_data, degree)
        y_grid = np.exp(np.polyval(z, x_grid))
        if increase:
            y_grid += y_min - epsilon
        else:
            y_grid = y_max + epsilon - y_grid
        return y_grid

    @staticmethod
    def inverse_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
                    degree: int = 1, y_lower_bound: float = 0.0) -> np.ndarray:
        fit_data = 1 / (y - y_lower_bound)
        z = np.polyfit(x, fit_data, degree)
        y_grid = 1 / (np.polyval(z, x_grid)) + y_lower_bound
        return y_grid

    @staticmethod
    def monotone(x: np.ndarray, increase: bool = True) -> np.ndarray:
        temp = 0.0
        y = np.copy(x)
        if increase:
            temp = min(x)
        else:
            temp = max(x)
        for i in range(len(x)):
            if ((increase and x[i] < temp) or (not increase and x[i] > temp)):
                y[i] = temp
            else:
                temp = x[i]
        return y

    @staticmethod
    def gaussian_kde(x: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        kde_func = stats.gaussian_kde(x)
        y_grid = kde_func(x_grid)
        return y_grid

    @staticmethod
    def interp_fit(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, interp_num: int = 20) -> np.ndarray:
        func = UnivariateSpline(x, y, s=interp_num)
        y_grid = func(x_grid)
        return y_grid
