# -*- coding: utf-8 -*-

import argparse

from PIL import Image
from collections import OrderedDict
# from scipy.io import savemat

from utils.data_utils import *
from utils.model_utils import validate, retrain_with_loader, target_class_proportion
from utils.main_utils import *

from config.perturb_config import get_perturb_params

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader',
                        dest='dataloader', default='valid')
    parser.add_argument('--data_dir', dest='data_dir', default='./data/')
    parser.add_argument('--result_dir', dest='result_dir',
                        default='./result/')
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    parser.add_argument('-b', '--batch_size',
                        dest='batch_size', default=16, type=int)

    parser.add_argument('--original', action='store_true',
                        dest='original', default=False)
    parser.add_argument('--alpha', dest='alpha', default=None, type=float)
    parser.add_argument('--org_alpha', dest='org_alpha', default=None, type=float)

    parser.add_argument('--mark_path', dest='mark_path',
                        default='./data/mark/apple_white.png')
    parser.add_argument('--mark_height',
                        dest='mark_height', default=0, type=int)
    parser.add_argument('--mark_width',
                        dest='mark_width', default=0, type=int)
    parser.add_argument('--mark_height_ratio',
                        dest='mark_height_ratio', default=1, type=float)
    parser.add_argument('--mark_width_ratio',
                        dest='mark_width_ratio', default=1, type=float)
    parser.add_argument('--mark_height_offset',
                        dest='mark_height_offset', default=None, type=int)
    parser.add_argument('--mark_width_offset',
                        dest='mark_width_offset', default=None, type=int)
    parser.add_argument('--preprocess_layer', dest='preprocess_layer',
                        default='features')

    parser.add_argument('-m', '--model_name', dest='model_name', default=None)
    parser.add_argument('--layer', dest='layer', default=None, type=int)

    parser.add_argument('-n', '--test_num',
                        dest='test_num', default=100, type=int)
    parser.add_argument('-e', '--epoch',
                        dest='epoch', default=20, type=int)
    parser.add_argument('-t', '--target_class',
                        dest='target_class', default=0, type=int)

    parser.add_argument('-i', '--input_method',
                        dest='input_method', default='watermark')
    parser.add_argument('--mode',
                        dest='mode', default='white')

    parser.add_argument('--learning_rate',
                        dest='learning_rate', default=0.015, type=float)
    parser.add_argument('--retrain_epoch',
                        dest='retrain_epoch', default=1, type=int)

    parser.add_argument('--output',
                        dest='output', default=0, type=int)

    args = parser.parse_args()

    params, DATA_PREFIX, model_name = parse_arguments(args)

    # ------------------------------------------------------------------------ #

    _model, model = load_model(args, params, model_name)
    criterion = _model.define_criterion()
    optimizer = _model.define_optimizer(lr=args.learning_rate)
    print(_model.name)

    _model.eval()
    _model.load_pretrained_weights()

    _perturb = load_perturb(args,  get_perturb_params(
        args.input_method, args.dataset, mode=args.mode), mode=args.mode, output=args.output)

    if args.alpha is not None:
        _perturb.alpha = args.alpha
    if args.org_alpha is None:
        args.org_alpha = _perturb.alpha

    trainloader = load_data(DATA_PREFIX, args.dataset, 'train')
    validloader = load_data(DATA_PREFIX, args.dataset, 'valid')
    testloader = load_data(DATA_PREFIX,
                           args.dataset, 'test', batch_size=args.batch_size)

    # validate(validloader, _model, criterion, params['num_classes'])
    # validate(testloader, _model, criterion, params['num_classes'])

    adv_result_dir = args.result_dir + '%s/%s/' % (args.dataset, _model.name)
    adv_img_dir = adv_img_dir = adv_result_dir + 'adv_img/%s/' % args.input_method

    adv_input_dir = adv_result_dir + 'adv_input/'
    adv_model_dir = adv_result_dir + 'adv_model/'

    if not os.path.exists(adv_result_dir):
        os.makedirs(adv_result_dir)

    # ------------------------------------------------------------------------ #
    shape = torch.tensor([0])
    for i, (Xi, Yi) in enumerate(trainloader):
        shape = to_tensor(to_tensor(Xi).shape)
        break
    mark = read_img_as_tensor(args.mark_path)
    org_mark_height = float(mark.shape[-2])
    org_mark_width = float(mark.shape[-1])
    if args.mark_height == 0 and args.mark_width == 0:
        args.mark_height = int(args.mark_height_ratio*float(shape[-2]))
        args.mark_width = int(args.mark_width_ratio*float(shape[-1]))
    assert args.mark_height != 0 and args.mark_width != 0
    if args.mark_height_offset is None and args.mark_height_offset is None:
        args.mark_height_offset = shape[-2]-args.mark_height
        args.mark_width_offset = shape[-1]-args.mark_width
    assert args.mark_height_offset is not None and args.mark_height_offset is not None

    mark = Image.fromarray(to_numpy(float2byte(mark)))
    mark = mark.resize((args.mark_width, args.mark_height), Image.ANTIALIAS)
    mark = byte2float(to_numpy(mark))
    print('Original Mark Shape: ', mark.shape)

    _file = 'a%f' % args.org_alpha + '_' + 'n%d' % args.test_num + '_' + 'e%d' % args.epoch + '_' +\
        't%d' % args.target_class + '_'+'lr%f' % args.learning_rate + '_'+'re%d' % args.retrain_epoch+'_' +\
        'mark(%d,%d)' % (args.mark_height, args.mark_width)+'_' +\
        'offset(%d,%d)' % (args.mark_height_offset, args.mark_width_offset)\
        + '_' + args.mode
    if args.original:
        _file += '_original'
    print(_file)
    _model.load_pretrained_weights(adv_result_dir+_file+'.pth')

    advmark = Image.open(adv_result_dir+_file+'.png')
    advmark = byte2float(to_numpy(advmark))

    advmark, mask = _perturb.mask_mark(
        advmark, shape, args.mark_height_offset, args.mark_width_offset, transparent='auto')
    print('Present Mark Shape: ', advmark.shape)
    print('Mask Shape: ', advmark.shape)
    save_tensor_as_img(adv_img_dir+'mark.png', advmark)

    for i, (Xi, Yi) in enumerate(testloader):
        X = to_tensor(Xi).detach()
        print("Original shape: ", X.shape)
        X2 = _perturb.add_mark(X, advmark, mask, _perturb.alpha)
        save_tensor_as_img(adv_img_dir+'%d.png' % i, X2[10])
        break
