'''
Author: your name
Date: 2020-09-30 22:13:19
LastEditTime: 2020-09-30 22:13:19
LastEditors: your name
Description: In User Settings Edit
FilePath: /xs/Trojan-Zoo/trojanzoo/defense/__init__.py
'''
# -*- coding: utf-8 -*-

from .defense import Defense
from .defense_backdoor import Defense_Backdoor
from .adv import *
from .backdoor import *

class_dict = {
    'defense': 'Defense',
    'defense_backdoor': 'Defense_Backdoor',

    'advmind': 'AdvMind',
    'curvature': 'Curvature',
    'grad_train': 'Grad_Train',
    'adv_train': 'Adv_Train',

    'neural_cleanse': 'Neural_Cleanse',
    'tabor': 'TABOR',
    'strip': 'STRIP',
    'abs': 'ABS',
    'activation_clustering': 'Activation_Clustering',
    'fine_pruning': 'Fine_Pruning',
    'deep_inspect': 'Deep_Inspect',
    'spectral_signature': 'Spectral_Signature',
    'neuron_inspect': 'Neuron_Inspect',
    'image_transform': 'Image_Transform',
    'magnet': 'MagNet',
    "model_agnostic": "Model_Agnostic"
}
