#!/usr/bin/env python3

from .version import __version__ as internal_version
from torch.torch_version import TorchVersion

from trojanzoo import environ as environ
from trojanzoo import datasets as datasets
from trojanzoo import models as models
from trojanzoo import trainer as trainer

from trojanzoo.utils.module import summary

__all__ = ['summary']
__version__ = TorchVersion(internal_version)
