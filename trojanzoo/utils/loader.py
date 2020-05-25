# -*- coding: utf-8 -*-

# from trojanzoo.datasets import Dataset
# from trojanzoo.models import Model
# from trojanzoo.attack import Attack


def get_module(module_class, module_name, **kwargs):
    pkg = __import__('package.'+module_class, fromlist=['class_dict'])
    class_dict = getattr(pkg, 'class_dict')
    class_name = class_dict[module_name]
    _class = getattr(pkg, class_name)

    return _class(**kwargs)


# def get_dataset(dataset_name, **kwargs):
#     return get_module('dataset', dataset_name, **kwargs)


# def get_model(model_name, **kwargs):
#     model_name, kwargs = model_func(model_name, **kwargs)
#     return get_module('model', model_name, **kwargs)


# def get_attack(attack_name, **kwargs):
#     return get_module('attack', attack_name, **kwargs)


# def model_func(model_name, **kwargs):
#     model_name, layer = split_name(model_name, layer=None, default_layer=None)
#     if model_name is None:
#         if 'dataset' in kwargs.keys():
#             model_name = kwargs['dataset'].default_model
#         else:
#             raise ValueError('model name is None!')
#     if layer is not None:
#         if 'layer' in kwargs.keys():
#             if kwargs['layer'] is not None:
#                 raise ValueError()
#         kwargs['layer'] = layer
#     return model_name, kwargs
