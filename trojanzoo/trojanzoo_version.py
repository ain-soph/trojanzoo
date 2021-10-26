#!/usr/bin/env python3

from .version import __version__ as internal_version
from torch.torch_version import TorchVersion

__version__ = TorchVersion(internal_version)
