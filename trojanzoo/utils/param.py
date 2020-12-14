# -*- coding: utf-8 -*-


class Module(object):

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        args = list(args)
        args.append(kwargs)
        for module in args:
            self._update(module)
        return self

    def _update(self, module: dict):
        if module is None:
            return self
        for key, value in module.items():
            if value is None:
                continue
            if not isinstance(value, Module) and not isinstance(value, dict):
                self[key] = value
            elif (not isinstance(self[key], Module) and not isinstance(value, dict)) \
                    or key not in self.keys():
                value_copy = self.__class__(value)
                self[key] = value_copy
            else:
                if not isinstance(self[key], Module):
                    self[key] = self.__class__(self[key])
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

    # def copy(self):
    #     return self.__class__(self)

    def clear(self):
        for item in list(self.keys()):
            delattr(self, item)
        return self

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
