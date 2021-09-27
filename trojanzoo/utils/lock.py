#!/usr/bin/env python3

# https://github.com/n-gao/pytorch-kfac/blob/master/torch_kfac/utils/lock.py

from contextlib import contextmanager


class Lock:
    r"""
    A boolean lock class used for contextmanager. It's used in :class:`~trojanzoo.utils.fim.kfac.KFAC` to avoid auxiliary computation operations.

    * :attr:`__entered` specifies the current boolean status of the lock.

    Examples:
        >>> track = Lock()
        >>> print(bool(track))
        False
        >>> with track():
        >>>     print(bool(track))
        True
        >>> print(bool(track))
        False
    """

    def __init__(self) -> None:
        self.__entered: bool = False

    @contextmanager
    def __call__(self) -> None:
        assert not self.__entered
        try:
            self.__entered = True
            yield
        finally:
            self.__entered = False

    def __bool__(self) -> bool:
        return self.__entered

    def enable(self):
        """Set lock boolean value as `True`. It's used together with `Lock.disable()` when contextmanager is not suitable for the case."""
        self.__entered = True

    def disable(self):
        """Set lock boolean value as `False`. It's used together with `Lock.enable()` when contextmanager is not suitable for the case."""
        self.__entered = False
