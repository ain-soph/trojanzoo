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

    See Also:
        https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Args:
        window_size (int): The :attr:`maxlen` of :class:`~collections.deque`.
        fmt (str): The format pattern of ``str(self)``.

    Attributes:
        deque (~collections.deque): The unique data series.
        count (int): The amount of data.
        total (float): The sum of all data.
        fmt (str): The string pattern.

        median (float): The median of :attr:`deque`.
        avg (float): The avg of :attr:`deque`.
        global_avg (float):
          :math:`\frac{\text{total}}{\text{count}}`
        max (float): The max of :attr:`deque`.
        min (float): The min of :attr:`deque`.
        value (float): The last value of :attr:`deque`.
    """

    def __init__(self, name: str = '', window_size: int = None, fmt: str = '{global_avg:.3f}'):
        self.name = name
        self.deque = deque(maxlen=window_size)
        self.count: int = 0
        self.total: float = 0.0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> 'SmoothedValue':
        r"""Update :attr:`n` pieces of data with same :attr:`value`.

        .. code-block:: python

            self.deque.append(value)
            self.total += value * n
            self.count += n

        Args:
            value (float): the value to update.
            n (int): the number of data with same :attr:`value`.

        Returns:
            SmoothedValue: return ``self`` for stream usage.
        """
        self.deque.append(value)
        self.total += value * n
        self.count += n
        return self

    def update_list(self, value_list: list[float]) -> 'SmoothedValue':
        r"""Update :attr:`value_list`.

        .. code-block:: python

            for value in value_list:
                self.deque.append(value)
                self.total += value
            self.count += len(value_list)

        Args:
            value_list (list[float]): the value list to update.

        Returns:
            SmoothedValue: return ``self`` for stream usage.
        """
        for value in value_list:
            self.deque.append(value)
            self.total += value
        self.count += len(value_list)
        return self

    def reset(self) -> 'SmoothedValue':
        r"""Reset ``deque``, ``count`` and ``total`` to be empty.

        Returns:
            SmoothedValue: return ``self`` for stream usage.
        """
        self.deque = deque(maxlen=self.deque.maxlen)
        self.count = 0
        self.total = 0.0
        return self

    def synchronize_between_processes(self):
        r"""
        Warning:
            Does NOT synchronize the deque!
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
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
            name=self.name,
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
    r"""
    See Also:
        https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Args:
        delimiter (str): The delimiter to join different meter strings.
            Defaults to ``''``.
        meter_length (int): The minimum length for each meter.
            Defaults to ``20``.
        indent (int): The space indent for the entire string.
            Defaults to ``0``.

    Attributes:
        meters (dict[str, SmoothedValue]): The meter dict.
        delimiter (str): The delimiter to join different meter strings.
        meter_length (int): The minimum length for each meter.
        indent (int): The space indent for the entire string.
    """

    def __init__(self, delimiter: str = '',
                 meter_length: int = 20, indent: int = 0):
        self.meters: defaultdict[str, SmoothedValue] \
            = defaultdict(SmoothedValue)
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

    def log_every(self, iterable: Iterable[_T], header: str = '',
                  total: int = None, print_freq: int = 0,
                  indent: int = None) -> Generator[_T, None, None]:
        r"""Wrap an :class:`collections.abc.Iterable` with formatted outputs.

        * Middle Output:
          ``[ current / total ] str(self) {iter_time} {data_time} {memory}``
        * Final Output
          ``{header} str(self) {total_time} {iter_time} {data_time} {memory}``

        Args:
            iterable (~collections.abc.Iterable): The raw iterator.
            header (str): The header string for final output.
                Defaults to ``''``.
            total (int): The length of iterable,
                which is used to generate ``[ current / total ]``
                in middle output.
                If ``None``, use ``len(iterable)`` if possible.
                If not possible, the middle header will be hidden.
                Defaults to ``None``.
            print_freq (int): Middle output during iteration
                when ``current % print_freq == 0``.
                Defaults to ``0`` (never).
            indent (int): The space indent for the entire string.
                if ``None``, use ``self.indent``.
                Defaults to ``None``.

        :Example:
            .. seealso:: :func:`trojanzoo.utils.train.train()`
        """
        indent = indent if indent is not None else self.indent
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                pass
        i = 0
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        memory = SmoothedValue(fmt='{max:d}')
        MB = 1 << 20

        end = time.time()
        start_time = time.time()
        for i, obj in enumerate(iterable):
            cur_data_time = time.time() - end
            data_time.update(cur_data_time)
            yield obj
            cur_iter_time = time.time() - end
            iter_time.update(cur_iter_time)
            if torch.cuda.is_available():
                cur_memory = torch.cuda.max_memory_allocated() / MB
                memory.update(cur_memory)
            if print_freq and i % print_freq == 0:
                middle_header = '' if total is None else output_iter(i, total)
                length = max(len(remove_ansi(header)) - 10, 0)
                middle_header = middle_header.ljust(
                    length + get_ansi_len(middle_header))
                log_msg = self.delimiter.join([middle_header, str(self)])
                if env['verbose'] > 1:
                    iter_time_pattern = '{green}iter{reset}: {iter_time:.4f} s'
                    data_time_pattern = '{green}data{reset}: {data_time:.4f} s'
                    iter_time_str = iter_time_pattern.format(
                        iter_time=cur_iter_time, **ansi)
                    data_time_str = data_time_pattern.format(
                        data_time=cur_data_time, **ansi)
                    iter_time_str = iter_time_str.ljust(
                        self.meter_length + get_ansi_len(iter_time_str))
                    data_time_str = data_time_str.ljust(
                        self.meter_length + get_ansi_len(data_time_str))
                    log_msg = self.delimiter.join(
                        [log_msg, iter_time_str, data_time_str])
                if env['verbose'] > 2 and torch.cuda.is_available():
                    memory_str = '{green}memory{reset}: {memory:d} MB'.format(
                        memory=cur_memory, **ansi)
                    memory_str = memory_str.ljust(
                        self.meter_length + get_ansi_len(memory_str))
                    log_msg = self.delimiter.join([log_msg, memory_str])
                prints(log_msg, indent=indent + 10)
            end = time.time()
        self.synchronize_between_processes()
        total_time = time.time() - start_time
        total_time = str(datetime.timedelta(seconds=int(total_time)))

        total_time_str: str = '{green}time{reset}: {time}'.format(
            time=total_time, **ansi)
        total_time_str = total_time_str.ljust(
            self.meter_length + get_ansi_len(total_time_str))
        log_msg = self.delimiter.join([header, str(self), total_time_str])
        if env['verbose'] > 1:
            iter_time_str: str = '{green}iter{reset}: {iter_time} s'.format(
                iter_time=str(iter_time), **ansi)
            data_time_str: str = '{green}data{reset}: {data_time} s'.format(
                data_time=str(data_time), **ansi)
            iter_time_str = iter_time_str.ljust(
                self.meter_length + get_ansi_len(iter_time_str))
            data_time_str = data_time_str.ljust(
                self.meter_length + get_ansi_len(data_time_str))
            log_msg = self.delimiter.join(
                [log_msg, iter_time_str, data_time_str])
        if env['verbose'] > 2 and torch.cuda.is_available():
            memory_str: str = '{green}memory{reset}: {memory} MB'.format(
                memory=str(memory), **ansi)
            memory_str = memory_str.ljust(
                self.meter_length + get_ansi_len(memory_str))
            log_msg = self.delimiter.join([log_msg, memory_str])
        prints(log_msg, indent=indent)


class AverageMeter:
    r"""Computes and stores the average and current value.

    See Also:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Note:
        It is recommended to use :class:`SmoothedValue` instead.
    """

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
