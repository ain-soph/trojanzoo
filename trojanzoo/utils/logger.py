#!/usr/bin/env python3

from .environ import env
from .output import ansi, output_iter, prints, get_ansi_len, remove_ansi

import torch
import torch.distributed as dist
from collections import defaultdict, deque
import datetime
import time


from typing import Generator, Iterable, TypeVar    # TODO: python 3.10
_T = TypeVar("_T")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = None, fmt: str = '{global_avg:.3f}'):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def update_list(self, value_list: list[float]):
        for value in value_list:
            self.deque.append(value)
            self.total += value
        self.count += len(value_list)

    def reset(self):
        self.deque = deque(maxlen=self.deque.maxlen)
        self.count = 0
        self.total = 0.0

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = float(t[1])

    @property
    def median(self) -> float:
        try:
            d = torch.tensor(list(self.deque))
            return d.median().item()
        except Exception:
            return 0.0

    @property
    def avg(self) -> float:
        try:
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            if len(d) == 0:
                return 0.0
            return d.mean().item()
        except Exception:
            return 0.0

    @property
    def global_avg(self) -> float:
        try:
            return self.total / self.count
        except Exception:
            return 0.0

    @property
    def max(self) -> float:
        try:
            return max(self.deque)
        except Exception:
            return 0.0

    @property
    def min(self) -> float:
        try:
            return min(self.deque)
        except Exception:
            return 0.0

    @property
    def value(self) -> float:
        try:
            return self.deque[-1]
        except Exception:
            return 0.0

    def __str__(self):
        return self.fmt.format(
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


class MetricLogger(object):
    def __init__(self, delimiter: str = '', meter_length: int = 20, indent: int = 0):
        self.meters: defaultdict[str, SmoothedValue] = defaultdict(SmoothedValue)
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
                _str = '{green}{}{reset}: {}'.format(name, str(meter)[:5], **ansi)
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
        memory = SmoothedValue(fmt='{max:.0f}')  # Memory is measured by Max value
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
                middle_header = middle_header.ljust(length + get_ansi_len(middle_header))
                log_msg = self.delimiter.join([middle_header, str(self)])
                if env['verbose'] > 1:
                    iter_time_str = '{green}iter{reset}: {iter_time} s'.format(iter_time=str(iter_time), **ansi)
                    data_time_str = '{green}data{reset}: {data_time} s'.format(data_time=str(data_time), **ansi)
                    iter_time_str = iter_time_str.ljust(self.meter_length + get_ansi_len(iter_time_str))
                    data_time_str = data_time_str.ljust(self.meter_length + get_ansi_len(data_time_str))
                    log_msg = self.delimiter.join([log_msg, iter_time_str, data_time_str])
                if env['verbose'] > 2 and torch.cuda.is_available():
                    memory_str = '{green}memory{reset}: {memory} MB'.format(memory=str(memory), **ansi)
                    memory_str = memory_str.ljust(self.meter_length + get_ansi_len(memory_str))
                    log_msg = self.delimiter.join([log_msg, memory_str])
                prints(log_msg, indent=indent + 10)
            end = time.time()
        self.synchronize_between_processes()
        total_time = time.time() - start_time
        total_time = str(datetime.timedelta(seconds=int(total_time)))

        total_time_str = '{green}time{reset}: {time}'.format(time=total_time, **ansi)
        total_time_str = total_time_str.ljust(self.meter_length + get_ansi_len(total_time_str))
        log_msg = self.delimiter.join([header, str(self), total_time_str])
        if env['verbose'] > 1:
            iter_time_str = '{green}iter{reset}: {iter_time} s'.format(iter_time=str(iter_time), **ansi)
            data_time_str = '{green}data{reset}: {data_time} s'.format(data_time=str(data_time), **ansi)
            iter_time_str = iter_time_str.ljust(self.meter_length + get_ansi_len(iter_time_str))
            data_time_str = data_time_str.ljust(self.meter_length + get_ansi_len(data_time_str))
            log_msg = self.delimiter.join([log_msg, iter_time_str, data_time_str])
        if env['verbose'] > 2 and torch.cuda.is_available():
            memory_str = '{green}memory{reset}: {memory} MB'.format(memory=str(memory), **ansi)
            memory_str = memory_str.ljust(self.meter_length + get_ansi_len(memory_str))
            log_msg = self.delimiter.join([log_msg, memory_str])
        prints(log_msg, indent=indent)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
