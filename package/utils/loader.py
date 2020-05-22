# -*- coding: utf-8 -*-

from package.utils.model import split_name


def get_module(module_class_name, module_name, **kwargs):
    if module_class_name == 'model':
        module_name, kwargs = model_func(module_name, **kwargs)
    pkg = __import__('package.'+module_class_name, fromlist=['class_dict'])
    class_dict = getattr(pkg, 'class_dict')
    class_name = class_dict[module_name]
    _class = getattr(pkg, class_name)

    return _class(**kwargs)


def get_dataset(dataset_name, **kwargs):
    return get_module('dataset', dataset_name, **kwargs)


def get_model(model_name, **kwargs):
    return get_module('model', model_name, **kwargs)


def get_perturb(perturb_name, **kwargs):
    return get_module('perturb', perturb_name, **kwargs)


def model_func(model_name, **kwargs):
    model_name, layer = split_name(model_name, layer=None, default_layer=None)
    if model_name is None:
        if 'dataset' in kwargs.keys():
            model_name = kwargs['dataset'].default_model
        else:
            raise ValueError('model name is None!')
    if layer is not None:
        if 'layer' in kwargs.keys():
            if kwargs['layer'] is not None:
                raise ValueError()
        kwargs['layer'] = layer
    return model_name, kwargs

# from config.config import get_model_name, name2class, get_data_params

# from utils.data_utils import save_tensor_as_img
# from imports.universal import *
# from utils.utils import *


# def load_model(args, pos='model', **kwargs):
#     params = parse_arguments(args)
#     name = get_model_name(args.model_name)
#     _class = getattr(__import__(pos + '.' + name,
#                                 fromlist=[name2class(name)]), name2class(name))
#     _model = _class(name=args.model_name, data_dir=args.data_dir, dataset=args.dataset,
#                     layer=args.layer, **params, **kwargs)
#     _model.eval()
#     if torch.cuda.is_available():
#         _model = _model.cuda()
#     model = parallel_model(_model)

#     return _model, model


# def load_perturb(args):

#     _class = getattr(__import__('perturb.' + args.input_method,
#                                 fromlist=[name2class(args.input_method)]),
#                      name2class(args.input_method))
#     _perturb = _class(args)
#     return _perturb


# def remove_misclassify_inside_batch(X, Y, org_result, i, test_idx, batch_size):
#     _, org_classification = org_result.max(-1)
#     _X = []
#     _Y = []
#     _idx = []
#     for idx in range(batch_size):
#         if org_classification[idx] == Y[idx]:
#             _X.append(X[idx])
#             _Y.append(Y[idx])
#             _idx.append(i * batch_size + idx)
#             test_idx.append(i * batch_size + idx)
#     X = to_tensor(_X)
#     Y = to_tensor(_Y)
#     _idx = to_numpy(_idx)
#     org_result = org_result[_idx - i * batch_size]
#     print('Current_Batch_Size: %d -> %d' % (batch_size, Y.shape[0]))

#     return X, Y, org_result, test_idx, _idx


# def after_attack(args, _dict, succ_idx, test_idx, iter_list, _model, model, X, adv, target, cur_iter, _idx, org_result, adv_img_dir=None, save_model=False):
#     batch_size = _idx.shape[0]
#     if args.output > 0:
#         org_confidence, org_classification = org_result.max(-1)
#     adv_result = F.softmax(model(adv))
#     adv_confidence, adv_classification = adv_result.max(1)

#     for idx in range(batch_size):
#         if adv_classification[idx] != target[idx] or adv_confidence[idx] < args.stop_confidence:
#             continue
#         _dict[str(_idx[idx])] = to_numpy(adv[idx].unsqueeze(0))
#         succ_idx.append(_idx[idx])
#         iter_list.append(cur_iter)
#         print('succ idx: %d, iter: %d, target:%s' %
#               (_idx[idx], cur_iter, target[idx]))

#         if args.save_img:
#             if adv_img_dir is None:
#                 adv_result_dir = args.data_dir + \
#                     'result/%s/%s/' % (args.dataset, _model.name)
#                 adv_img_dir = adv_result_dir + 'adv_img/'
#             _path = adv_img_dir + '%d_org.jpg' % (_idx[idx])
#             save_tensor_as_img(_path, X[idx])
#             _path = adv_img_dir + '%d_adv.jpg' % (_idx[idx])
#             save_tensor_as_img(_path, adv[idx])
#             noise_path = adv_img_dir + '%d_noise.jpg' % (_idx[idx])
#             save_tensor_as_img(noise_path, adv[idx]-X[idx])

#         if args.output > 0:
#             print('Original:')
#             print(org_result[idx])
#             print(org_classification[idx])
#             print(org_confidence[idx])
#             print('Adversarial:')
#             print(adv_result[idx])
#             print(adv_classification[idx])
#             print(adv_confidence[idx])
#             print('Iteration: ', cur_iter)
#             print(target[idx])
#             print()
#     print()
