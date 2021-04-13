#!/usr/bin/env python3

from trojanvision.models.imagemodel import ImageModel

from .natsbench import NATSbench

from .darts import DARTS
from .enas import ENAS
from .lanet import LaNet
from .mnasnet import MNASNet
from .pnasnet import PNASNet
from .proxylessnas import ProxylessNAS

__all__ = ['NATSbench', 'DARTS', 'ENAS', 'LaNet', 'MNASNet', 'PNASNet', 'ProxylessNAS']

class_dict: dict[str, type[ImageModel]] = {
    'natsbench': NATSbench,
    'darts': DARTS,
    'enas': ENAS,
    'lanet': LaNet,
    'mnasnet': MNASNet,
    'pnasnet': PNASNet,
    'proxylessnas': ProxylessNAS,
}
