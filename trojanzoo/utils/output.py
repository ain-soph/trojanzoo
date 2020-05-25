# -*- coding: utf-8 -*-

import torch


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


def output_memory(indent=0, device=None, full=False, **kwargs):
    if full:
        prints(torch.cuda.memory_summary(device=device, **kwargs))
    else:
        prints('memory allocated: '.ljust(20),
               bytes2size(torch.cuda.memory_allocated(device=device)), indent=indent)
        prints('memory cached: '.ljust(20),
               bytes2size(torch.cuda.memory_cached(device=device)), indent=indent)


def bytes2size(_bytes):
    if _bytes < 2*1024:
        return '%d bytes' % _bytes
    elif _bytes < 2*1024*1024:
        return '%.3f KB' % (float(_bytes)/1024)
    elif _bytes < 2*1024*1024*1024:
        return '%.3f MB' % (float(_bytes)/1024/1024)
    else:
        return '%.3f GB' % (float(_bytes)/1024/1024/1024)
