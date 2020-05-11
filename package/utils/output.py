# -*- coding: utf-8 -*-
import torch
from .utils import bytes2size


def prints(*args, indent=0, **kwargs):
    assert indent >= 0
    print(' '*indent, end='')
    print(*args, **kwargs)


def output_iter(_iter, iteration=None):
    if iteration is None:
        return '[ ' + str(_iter).center(max(3, len(str(_iter))), ' ') + ' ]'
    else:
        length = len(str(iteration))
        return '[ ' + str(_iter).ljust(length, ' ') + ' / %d ]' % iteration


def output_memory(indent=0):
    prints('memory allocated: '.ljust(20),
           bytes2size(torch.cuda.memory_allocated()), indent=indent)
    prints('memory cached: '.ljust(20),
           bytes2size(torch.cuda.memory_cached()), indent=indent)


