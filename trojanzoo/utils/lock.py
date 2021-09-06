#!/usr/bin/env python3

# https://github.com/n-gao/pytorch-kfac/blob/master/torch_kfac/utils/lock.py

from contextlib import contextmanager


class Lock:
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
        self.__entered = True

    def disable(self):
        self.__entered = False
