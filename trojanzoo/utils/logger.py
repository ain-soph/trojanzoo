#!/usr/bin/env python3

from .output import ansi, output_iter, prints, get_ansi_len, remove_ansi
from trojanzoo.environ import env

import torch
import torch.distributed as dist
from collections import defaultdict, deque
import datetime
import time


from typing import Generator, Iterable, TypeVar    # TODO: python 3.10
_T = TypeVar("_T")

__all__ = ['SmoothedValue', 'MetricLogger', 'AverageMeter']


class SmoothedValue:
    r"""Track a series of values and provide access to smoothed values over a
    window or the global series average.

    Args:
        window_size (int): The :attr:`maxlen` of :class:`~collections.deque`.
        fmt (str): The format pattern of ``str(self)``.

    Attributes:
        __deque (~collections.deque): The data series.
        __total (float): The sum of all data.
        __count (int): The amount of data.
        __fmt (str): The string pattern.
    """

    def __init__(self, window_size: int = None, fmt: str = '{global_avg:.3f}'):
        self.__deque = deque(maxlen=window_size)
        self.__total = 0.0
        self.__count = 0
        self.__fmt = fmt

    def update(self, value: float, n: int = 1) -> 'SmoothedValue':
        r"""update :attr:`n` pieces of data with same :attr:`value`
        into :class:`~collections.deque`.

        Args:
            value (float): the value to update.
            n (int): the number of data with same :attr:`value`.

        Returns
        -------
            self: :class:`SmoothedValue`
                return self for stream usage.
        """
        self.__deque.append(value)
        self.__count += n
        self.__total += value * n
        return self

    def update_list(self, value_list: list[float]) -> 'SmoothedValue':
        for value in value_list:
            self.__deque.append(value)
            self.__total += value
        self.__count += len(value_list)
        return self

    def reset(self):
        self.__deque = deque(maxlen=self.__deque.maxlen)
        self.__count = 0
        self.__total = 0.0

    def synchronize_between_processes(self):
        r"""
        Warning: does not synchronize the deque!
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([self.__count, self.__total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.__count = int(t[0])
        self.__total = float(t[1])

    @property
    def median(self) -> float:
        try:
            d = torch.tensor(list(self.__deque))
            return d.median().item()
        except Exception:
            return 0.0

    @property
    def avg(self) -> float:
        try:
            d = torch.tensor(list(self.__deque), dtype=torch.float32)
            if len(d) == 0:
                return 0.0
            return d.mean().item()
        except Exception:
            return 0.0

    @property
    def global_avg(self) -> float:
        try:
            return self.__total / self.__count
        except Exception:
            return 0.0

    @property
    def max(self) -> float:
        try:
            return max(self.__deque)
        except Exception:
            return 0.0

    @property
    def min(self) -> float:
        try:
            return min(self.__deque)
        except Exception:
            return 0.0

    @property
    def value(self) -> float:
        try:
            return self.__deque[-1]
        except Exception:
            return 0.0

    def __str__(self):
        return self.__fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            min=self.min,
            max=self.max,
            value=self.value)

    def __format__(self, format_spec: str) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()


class MetricLogger:
    def __init__(self, delimiter: str = '',
                 meter_length: int = 20, indent: int = 0):
        self.meters: defaultdict[str,
                                 SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.meter_length = meter_length
        self.indent = indent

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(float(v))

    def __getattr__(self, attr: str) -> float:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:   # TODO: use hasattr
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str: list[str] = []
        for name, meter in self.meters.items():
            _str = '{green}{}{reset}: {}'.format(name, str(meter), **ansi)
            max_length = self.meter_length + get_ansi_len(_str)
            if len(_str) > max_length:
                _str = '{green}{}{reset}: {}'.format(
                    name, str(meter)[:5], **ansi)
            _str = _str.ljust(max_length)
            loss_str.append(_str)
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable: Iterable[_T], header: str = None,
                  total: int = None, print_freq: int = 0,
                  indent: int = None) -> Generator[_T, None, None]:
        indent = indent if indent is not None else self.indent
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                pass
        i = 0
        if not header:
            header = ''
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # Memory is measured by Max value
        memory = SmoothedValue(fmt='{max:.0f}')
        MB = 1024.0 * 1024.0

        end = time.time()
        start_time = time.time()
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if torch.cuda.is_available():
                memory.update(torch.cuda.max_memory_allocated() / MB)
            if print_freq and i % print_freq == 0:
                middle_header = '' if total is None else output_iter(i, total)
                length = max(len(remove_ansi(header)) - 10, 0)
                middle_header = middle_header.ljust(
                    length + get_ansi_len(middle_header))
                log_msg = self.delimiter.join([middle_header, str(self)])
                if env['verbose'] > 1:
                    iter_time_str = '{green}iter{reset}: {iter_time} s'.format(
                        iter_time=str(iter_time), **ansi)
                    data_time_str = '{green}data{reset}: {data_time} s'.format(
                        data_time=str(data_time), **ansi)
                    iter_time_str = iter_time_str.ljust(
                        self.meter_length + get_ansi_len(iter_time_str))
                    data_time_str = data_time_str.ljust(
                        self.meter_length + get_ansi_len(data_time_str))
                    log_msg = self.delimiter.join(
                        [log_msg, iter_time_str, data_time_str])
                if env['verbose'] > 2 and torch.cuda.is_available():
                    memory_str = '{green}memory{reset}: {memory} MB'.format(
                        memory=str(memory), **ansi)
                    memory_str = memory_str.ljust(
                        self.meter_length + get_ansi_len(memory_str))
                    log_msg = self.delimiter.join([log_msg, memory_str])
                prints(log_msg, indent=indent + 10)
            end = time.time()
        self.synchronize_between_processes()
        total_time = time.time() - start_time
        total_time = str(datetime.timedelta(seconds=int(total_time)))

        total_time_str = '{green}time{reset}: {time}'.format(
            time=total_time, **ansi)
        total_time_str = total_time_str.ljust(
            self.meter_length + get_ansi_len(total_time_str))
        log_msg = self.delimiter.join([header, str(self), total_time_str])
        if env['verbose'] > 1:
            iter_time_str = '{green}iter{reset}: {iter_time} s'.format(
                iter_time=str(iter_time), **ansi)
            data_time_str = '{green}data{reset}: {data_time} s'.format(
                data_time=str(data_time), **ansi)
            iter_time_str = iter_time_str.ljust(
                self.meter_length + get_ansi_len(iter_time_str))
            data_time_str = data_time_str.ljust(
                self.meter_length + get_ansi_len(data_time_str))
            log_msg = self.delimiter.join(
                [log_msg, iter_time_str, data_time_str])
        if env['verbose'] > 2 and torch.cuda.is_available():
            memory_str = '{green}memory{reset}: {memory} MB'.format(
                memory=str(memory), **ansi)
            memory_str = memory_str.ljust(
                self.meter_length + get_ansi_len(memory_str))
            log_msg = self.delimiter.join([log_msg, memory_str])
        prints(log_msg, indent=indent)


class AverageMeter:
    r"""Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.__fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.__count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.__count += n
        self.avg = self.sum / self.__count

    def __str__(self):
        fmtstr = '{name} {val' + self.__fmt + '} ({avg' + self.__fmt + '})'
        return fmtstr.format(**self.__dict__)
