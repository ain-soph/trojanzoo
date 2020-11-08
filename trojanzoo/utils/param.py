# -*- coding: utf-8 -*-


class Module(object):

    def __init__(self, *args, **kwargs):
        self.add(*args, **kwargs)

    def add(self, *args, **kwargs):
        args = list(args)
        args.append(kwargs)
        for module in args:
            for key, value in module.items():
                if value is None:
                    continue
                if isinstance(value, dict) or isinstance(value, Module):
                    value = self.__class__(value)
                self.__setattr__(key, value)
        return self

    def update(self, module: dict):
        if module is None:
            return self
        if isinstance(module, dict):
            module = self.__class__(module)
        for key, value in module.items():
            if value is None:
                continue
            if key not in self.keys() or not isinstance(value, Module):
                if isinstance(value, Module):
                    value = value.copy()
                self[key] = value
            elif not isinstance(self[key], Module):
                if isinstance(value, Module):
                    value = value.copy()
                self[key] = value
            else:
                self[key].update(value)
        return self

    def remove_none(self):
        """Remove the parameters whose values are ``None``

        :return: ``self``
        :rtype: Module
        """
        for key in list(self.keys()):
            if self[key] is None:
                delattr(self, key)
        return self

    def summary(self, indent=0):
        print(' ' * indent, self)

    def copy(self):
        return self.__class__(self)

    def clear(self):
        for item in list(self.keys()):
            delattr(self, item)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()


class Param(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self['default']

    def __getitem__(self, key) -> Module:
        if key not in self.keys():
            if 'default' not in self.keys():
                return None
            key = 'default'
        return super().__getitem__(key)
