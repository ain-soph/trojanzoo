#!/usr/bin/env python3

from ...abstract import TrainingFiltering
from .activation_clustering import ActivationClustering
# from .spectral_signature import SpectralSignature

__all__ = ['ActivationClustering']   # 'SpectralSignature',

class_dict: dict[str, type[TrainingFiltering]] = {
    'activation_clustering': ActivationClustering,
    # 'spectral_signature': SpectralSignature,
}
