#!/usr/bin/env python3

from trojanzoo.utils.output import prints

from collections.abc import MutableMapping
from typing import Self, Generic, TypeVar

_KT = TypeVar("_KT")  # key type
_VT = TypeVar("_VT")  # value type


class Module(MutableMapping[str, _VT], Generic[_KT, _VT]):
    r"""A dict-like class which supports attribute-like view as well.

    Args:
        *args: Positional dict-like arguments.
            All keys will be merged together.
        **kwargs: Keyword arguments that compose a dict.
            All keys will be merged together.

    Attributes:
        _marker (str): The marker of the class,
            which is shown in ``str(self)``.
            Defaults to ``'M'``.
    """
    _marker: str = 'M'

    def __init__(self, *args: MutableMapping[str, _VT], **kwargs: _VT):
        self.__data: dict[str, _VT] = {}
        if len(args) == 1 and args[0] is None:
            return
        self.update(*args, **kwargs)

    def update(self, *args: MutableMapping[str, _VT], **kwargs: _VT) -> Self:
        r"""update values.

        Args:
            *args: Positional dict-like arguments.
                All keys will be merged together.
            **kwargs: Keyword arguments that compose a dict.
                All keys will be merged together.

        Returns:
            Module: return :attr:`self` for stream usage.
        """
        args_list = list(args)
        args_list.append(kwargs)
        for module in args_list:
            self._update(module)
        return self

    def _update(self, module: MutableMapping[str, _VT]) -> Self:
        for key, value in module.items():
            if value is None:
                continue
            if key in self.keys() and isinstance(self[key], Module):
                sub_module: Module = self[key]
                sub_module.update(value)
            elif key in self.keys() and isinstance(value, Module):
                self[key] = type(value)(self[key]).update(value)
            elif isinstance(value, Module):
                self[key] = value.copy()
            else:
                self[key] = value
        return self

    def remove_none(self) -> Self:
        r"""Remove the parameters whose values are ``None``.

        Returns:
            Module: return :attr:`self` for stream usage.
        """
        for key in self.__data.keys():
            if self.__data[key] is None:
                del self.__data[key]
        return self

    def copy(self):
        r"""Deepcopy of :attr:`self`.

        Returns:
            Module: return the deepcopy of :attr:`self`.
        """
        return type(self)(self)

    def clear(self) -> Self:
        r"""Remove all keys.

        Returns:
            Module: return :attr:`self` for stream usage.
        """
        self.__data = {}
        return self

    def keys(self):
        return self.__data.keys()

    def items(self):
        return self.__data.items()

    def __getattr__(self, name: str) -> _VT:
        if name.startswith('__'):
            return super().__getattr__(name)
        return self.__data[name]

    def __getitem__(self, k: str):
        return self.__data[k]

    def __setattr__(self, name: str, value: _VT):
        if '__data' in name:
            return super().__setattr__(name, value)
        self.__data[name] = value

    def __setitem__(self, k: str, v: _VT):
        self.__data[k] = v

    def __delattr__(self, name: str):
        del self.__data[name]

    def __delitem__(self, v: str):
        del self.__data[v]

    def __str__(self):
        return self._marker + self.__data.__str__()

    def __repr__(self):
        return self._marker + self.__data.__repr__()

    def __len__(self):
        return self.__data.__len__()

    def __iter__(self):
        return self.__data.__iter__()

    def summary(self, indent: int = 0):
        r"""Output information of :attr:`self`.

        Args:
            indent (int): The space indent for the entire string.
                Defaults to ``0``.
        """
        prints(self, indent=indent)


class Param(Module, Generic[_KT, _VT]):
    r"""A dict-like class to store parameters config that
    inherits :class:`Module` and further extends default values.
    You can view and set keys by attributes as well.

    Args:
        *args: Positional dict-like arguments.
            All keys will be merged together.
            If there is only 1 argument and no keyword argument,
            regard it as the default value.
        **kwargs: Keyword arguments that compose a dict.
            All keys will be merged together.

    Attributes:
        _marker (str): The marker of the class,
            which is shown in ``str(self)``.
            Defaults to ``'P'``.
        default (Any): The default value of unknown keys.
    """
    _marker = 'P'

    def update(self, *args: dict[str, _VT] | _VT, **kwargs: _VT) -> Self:
        if len(kwargs) == 0 and len(args) == 1 and \
                not isinstance(args[0], (dict, Module)):
            self.default = args[0]
            return self
        return super().update(*args, **kwargs)

    def _update(self, module: dict[str, _VT]) -> Self:
        for key, value in module.items():
            if key == 'default':
                self.default = value
        return super()._update(module)

    def remove_none(self) -> Self:
        for key in list(self.__data.keys()):
            if self.__data[key] is None and \
                    not (isinstance(key, str) and key == 'default'):
                del self.__data[key]
        return self

    def __getattr__(self, name: str) -> _VT:
        try:
            return super().__getattr__(name)
        except KeyError:
            if 'default' in self.keys():
                return self['default']
            raise

    def __getitem__(self, key: str) -> _VT:
        if key not in self.keys():
            key = 'default'
            if 'default' not in self.keys():
                print(self)
                raise KeyError(key)
        return super().__getitem__(key)

    def clear(self) -> Self:
        super().clear()
        self.default = None
        return self
