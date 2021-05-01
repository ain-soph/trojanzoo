#!/usr/bin/env python3

from .output import prints

from typing import TYPE_CHECKING
from typing import Generic, MutableMapping, TypeVar
_KT = TypeVar("_KT")  # Key type.
_VT = TypeVar("_VT")  # Value type.
if TYPE_CHECKING:
    pass


class Module(MutableMapping[_KT, _VT], Generic[_KT, _VT]):  # TODO: issue 3 why need Generic
    _marker = 'M'

    def __init__(self, *args: MutableMapping[_KT, _VT], **kwargs: _VT):
        self.__data: dict[_KT, _VT] = {}
        if len(args) == 1 and args[0] is None:
            return
        self.update(*args, **kwargs)

    def update(self, *args: MutableMapping[_KT, _VT], **kwargs: _VT):
        args: list = list(args)     # TODO: issue 2 pylance issue
        args.append(kwargs)
        for module in args:
            self._update(module)
        return self

    def _update(self, module: MutableMapping[_KT, _VT]):    # TODO: issue 4 Union[dict, Module]
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

    def remove_none(self):
        """Remove the parameters whose values are ``None``"""
        for key in self.__data.keys():
            if self.__data[key] is None:
                del self.__data[key]
        return self

    def copy(self):
        return type(self)(self)

    def clear(self):
        for item in self.keys():
            delattr(self, item)
        return self

    def keys(self):
        return self.__data.keys()

    def items(self):
        return self.__data.items()

    def __getattr__(self, name: _KT) -> _VT:
        if name == '_Module__data':
            return super().__getattr__(name)
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

    def __len__(self):
        return self.__data.__len__()

    def __iter__(self):
        return self.__data.__iter__()

    def summary(self, indent: int = 0):
        prints(self, indent=indent)


class Param(Module, Generic[_KT, _VT]):  # TODO: issue 3 why need Generic, Module[_KT, _VT]
    _marker = 'P'

    def update(self, *args: dict[_KT, _VT], **kwargs: _VT):
        if len(kwargs) == 0 and len(args) == 1 and not isinstance(args[0], (dict, Module)):
            self.default = args[0]
            return self
        return super().update(*args, **kwargs)

    def _update(self, module: dict[_KT, _VT]):
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
        except KeyError as e:
            if 'default' in self.keys():
                return self['default']
            raise e

    def __getitem__(self, key: str) -> _VT:
        if key not in self.keys():
            key = 'default'
            if 'default' not in self.keys():
                print(self)
                raise KeyError(key)
        return super().__getitem__(key)
