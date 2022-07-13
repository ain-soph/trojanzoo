#!/usr/bin/env python3

from ...abstract import BackdoorDefense
from .abs import ABS
from .deep_inspect import DeepInspect
from .neural_cleanse import NeuralCleanse
from .neuron_inspect import NeuronInspect
from .tabor import Tabor

__all__ = ['ABS', 'DeepInspect', 'NeuralCleanse', 'NeuronInspect', 'Tabor']

class_dict: dict[str, type[BackdoorDefense]] = {
    'abs': ABS,
    'deep_inspect': DeepInspect,
    'neural_cleanse': NeuralCleanse,
    'neuron_inspect': NeuronInspect,
    'tabor': Tabor,
}
