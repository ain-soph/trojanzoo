#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .output import prints
from types import GenericAlias
from typing import Any, Generic, MutableMapping, Iterator, TypeVar


# typing.pyi
_KT = TypeVar("_KT")  # Key type.
_VT = TypeVar("_VT")  # Value type.
_T_co = TypeVar("_T_co", covariant=True)  # Any type covariant containers.


class Module(MutableMapping[_KT, _VT], Generic[_KT, _VT]):
    _marker = 'M'

    def __class_getitem__(cls, item: Any) -> GenericAlias:
        return _VT

    def __init__(self, *args: list[dict[_KT, _VT]], **kwargs):
        self.__data: dict[_KT, _VT] = {}
        if len(args) == 1 and args[0] is None:
            return
        self.update(*args, **kwargs)

    def update(self, *args: list[dict[_KT, _VT]], **kwargs):
        args = list(args)
        args.append(kwargs)
        for module in args:
            self._update(module)
        return self

    def _update(self, module: dict[_KT, _VT]):    # TODO: Union[dict, Module]
        for key, value in module.items():
            if value is None:
                continue
            if key in self.keys() and isinstance(self[key], Module):
                self[key].update(value)     # TODO: linting problem
            elif key in self.keys() and isinstance(value, Module):
                self[key] = type(value)(self[key])
                self[key].update(value)
            elif isinstance(value, Module):
                self[key] = value.copy()
            else:
                self[key] = value
        return self

    def remove_none(self):
        """Remove the parameters whose values are ``None``

        :return: ``self``
        :rtype: Module
        """
        for key in list(self.__data.keys()):
            if self.__data[key] is None:
                del self.__data[key]
        return self

    def copy(self):
        return type(self)(self)

    def clear(self):
        for item in list(self.keys()):
            delattr(self, item)
        return self

    def keys(self):
        return self.__data.keys()

    def items(self):
        return self.__data.items()

    def __getattr__(self, name: _KT) -> _VT:
        if name == '_Module__data':
            super().__getattr__(name)
        return self.__data[name]

    def __getitem__(self, k: _KT):
        return self.__data[k]

    def __setattr__(self, name: _KT, value: _VT):
        if name == '_Module__data':
            return super().__setattr__(name, value)
        self.__data[name] = value

    def __setitem__(self, k: _KT, v: _VT):
        self.__data[k] = v

    def __delattr__(self, name: _KT):
        del self.__data[name]

    def __delitem__(self, v: _KT):
        del self.__data[v]

    def __str__(self):
        return self._marker + self.__data.__str__()

    def __repr__(self):
        return self._marker + self.__data.__repr__()

    def __len__(self) -> int:
        return self.__data.__len__()

    def __iter__(self) -> Iterator[_T_co]:
        return self.__data.__iter__()

    def summary(self, indent: int = 0):
        prints(self, indent=indent)


class Param(Module, Generic[_KT, _VT]):
    _marker = 'P'

    def update(self, *args: list[dict], **kwargs):
        if len(kwargs) == 0 and len(args) == 1 and not isinstance(args[0], dict) \
                and not isinstance(args[0], Module):
            self.default = args[0]
            return self
        return super().update(*args, **kwargs)

    def _update(self, module: dict[str, Any]):
        for key, value in module.items():
            if key == 'default':
                self.default = value
        super()._update(module)
        return self     # For linting purpose

    def remove_none(self):
        """Remove the parameters whose values are ``None``

        :return: ``self``
        :rtype: Module
        """
        for key in list(self.__data.keys()):
            if self.__data[key] is None and not (isinstance(key, str) and key == 'default'):
                del self.__data[key]
        return self

    def __getattr__(self, name: str) -> _VT:
        try:
            return super().__getattr__(name)
        except KeyError:    # TODO: or AttributeError better?
            return self['default']

    def __getitem__(self, key: str) -> _VT:
        if key not in self.keys():
            key = 'default'
            if 'default' not in self.keys():
                print(self)
                raise KeyError(key)
        return super().__getitem__(key)
