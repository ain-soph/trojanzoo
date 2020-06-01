# -*- coding: utf-8 -*-

from trojanzoo.utils import add_noise, empty_cache, repeat_to_batch
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.utils.model import split_name as func
from trojanzoo.utils.model import AverageMeter, CrossEntropy
from trojanzoo.dataset.dataset import Dataset

import types
from typing import Union

import os
import time
import datetime
from collections import OrderedDict
from collections.abc import Iterable
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from trojanzoo.config import Config
env = Config.env


class _Model(nn.Module):
    def __init__(self, num_classes: int = None, conv_depth=0, conv_dim=0, fc_depth=0, fc_dim=0, **kwargs):
        super().__init__()

        self.conv_depth = conv_depth
        self.conv_dim = conv_dim
        self.fc_depth = fc_depth
        self.fc_dim = fc_dim
        self.num_classes = num_classes

        self.features = self.define_features()   # feature extractor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # average pooling
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = self.define_classifier()  # classifier

    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)
    def forward(self, x):
        # if x.shape is (channels, height, width)
        # (channels, height, width) ==> (batch_size: 1, channels, height, width)
        x = self.get_final_fm(x)
        x = self.get_logits_from_fm(x)
        return x

    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x):
        return self.features(x)

    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_final_fm(self, x):
        x = self.get_fm(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    # get logits from feature map
    # input: (batch_size, [feature_map])
    # output: (batch_size, logits)
    def get_logits_from_fm(self, x):
        return self.classifier(x)

    def define_features(self, conv_depth: int = None, conv_dim: int = None):
        return nn.Identity()

    def define_classifier(self, num_classes: int = None, conv_dim: int = None, fc_depth: int = None, fc_dim: int = None):
        if fc_depth is None:
            fc_depth = self.fc_depth
        if self.fc_depth <= 0:
            return nn.Identity()
        if conv_dim is None:
            conv_dim = self.conv_dim
        if fc_dim is None:
            fc_dim = self.fc_dim
        if num_classes is None:
            num_classes = self.num_classes

        seq = []
        if self.fc_depth == 1:
            seq.append(('fc', nn.Linear(self.conv_dim, self.num_classes)))
        else:
            seq.append(('fc1', nn.Linear(self.conv_dim, self.fc_dim)))
            seq.append(('relu1', nn.ReLU()))
            seq.append(('dropout1', nn.Dropout()))
            for i in range(self.fc_depth-2):
                seq.append(
                    ('fc'+str(i+2), nn.Linear(self.fc_dim, self.fc_dim)))
                seq.append(('relu'+str(i+2), nn.ReLU()))
                seq.append(('dropout'+str(i+2), nn.Dropout()))
            seq.append(('fc'+str(self.fc_depth),
                        nn.Linear(self.fc_dim, self.num_classes)))
        return nn.Sequential(OrderedDict(seq))


class Model:

    def __init__(self, name='model', model_class=_Model, dataset: Dataset = None,
                 num_classes: int = None, loss_weights: torch.FloatTensor = None,
                 official=False, pretrain=False, prefix='', folder_path: str = None, **kwargs):
        self.name = name
        self.dataset = dataset
        self.prefix = prefix

        #------------Auto--------------#
        if dataset is not None:
            data_dir: str = env['data_dir']
            if isinstance(dataset, str):
                raise TypeError(dataset)
            if folder_path is None:
                folder_path = data_dir+dataset.data_type+'/'+dataset.name+'/model/'
            if num_classes is None:
                num_classes = dataset.num_classes
            if loss_weights is None:
                loss_weights = dataset.loss_weights
        self.num_classes = num_classes  # number of classes
        self.loss_weights = loss_weights

        self.folder_path = folder_path

        #------------------------------#
        self.criterion = self.define_criterion(loss_weights=loss_weights)
        self.softmax = nn.Softmax(dim=1)

        #-----------Temp---------------#
        # the location when loading pretrained weights using torch.load
        self._model = model_class(num_classes=num_classes, **kwargs)
        self.model = self.get_parallel()
        # load pretrained weights
        if official:
            self.load('official')
        if pretrain:
            self.load()
        if env['num_gpus']:
            self.cuda()
        self.eval()

    #----------------- Forward Operations ----------------------#

    def get_logits(self, _input, randomized_smooth=False, sigma=0.01, n=100, **kwargs):
        if randomized_smooth:
            _list = []
            for _ in range(n):
                _input_noise = add_noise(_input, std=sigma, detach=False)
                _list.append(self.model(_input_noise))
            return torch.stack(_list).mean(dim=0)
        else:
            return self.model(_input)

    def get_prob(self, _input, **kwargs):
        return self.softmax(self.get_logits(_input, **kwargs))

    def get_target_prob(self, _input, target, **kwargs):
        return self.get_prob(_input, **kwargs)[:, target]

    def get_class(self, _input, **kwargs):
        return self.get_logits(_input, **kwargs).argmax(dim=-1)

    def loss(self, _input, _label, **kwargs):
        return self.criterion(self(_input, **kwargs), _label)

    #--------------------------------------------------------#

    # Define the optimizer
    # and transfer to that tuning mode.
    # train_opt: 'full' or 'partial' (default: 'partial')
    # lr: (default: [full:2e-3, partial:2e-4])
    # optim_type: to be implemented
    #
    # return: optimizer

    def define_optimizer(self, lr: float = 0.1,
                         parameters: Union[str, Iterable] = 'full', optim_type: Union[str, type] = None,
                         lr_scheduler=True, step_size=30, **kwargs):

        if isinstance(parameters, str):
            if parameters == 'full':
                parameters = self._model.parameters()
            elif parameters == 'classifier':
                parameters = self._model.classifier.parameters()
            else:
                raise NotImplementedError(parameters)
        if not isinstance(parameters, Iterable):
            raise TypeError(type(parameters))

        parameters, param_copy = itertools.tee(parameters)
        self.activate_params(param_copy)

        if optim_type is None:
            optim_type = optim.SGD
        elif isinstance(optim_type, str):
            optim_type = getattr(optim, optim_type)
        assert isinstance(optim_type, type)

        if kwargs == {}:
            if optim_type == optim.SGD:
                kwargs = {'momentum': 0.9,
                          'weight_decay': 2e-4, 'nesterov': True}
        optimizer = optim_type(parameters, lr, **kwargs)
        _lr_scheduler = None
        if lr_scheduler:
            _lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.1)
            # optimizer = optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[150, 250], gamma=0.1)
        return optimizer, _lr_scheduler

    # define loss function
    # Cross Entropy
    def define_criterion(self, loss_weights: torch.FloatTensor = None):
        return nn.CrossEntropyLoss(weight=loss_weights)

    #-----------------------------Load & Save Model-------------------------------------------#

    # file_path: (default: '') if '', use the default path. Else if the path doesn't exist, quit.
    # full: (default: False) whether save feature extractor.
    # output: (default: False) whether output help information.
    def load(self, file_path: str = None, folder_path: str = None, prefix: str = None,
             features=True, map_location='default', verbose=False):
        if map_location is not None:
            if map_location == 'default':
                map_location = env['device']
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if prefix is None:
                prefix = self.prefix
            file_path = folder_path + self.name + prefix + '.pth'
        elif file_path == 'official':
            return self.load_official_weights()
        if os.path.exists(file_path):
            try:
                if features:
                    self._model.load_state_dict(
                        torch.load(file_path, map_location=map_location))
                else:
                    self._model.classifier.load_state_dict(
                        torch.load(file_path, map_location=map_location))
            except Exception as e:
                print('Model file path: ', file_path)
                raise e
        else:
            raise FileNotFoundError('Model file not exist: ', file_path)
        if verbose:
            print("Model {} loaded from: ".format(self.name), file_path)

    # file_path: (default: '') if '', use the default path.
    # full: (default: False) whether save feature extractor.
    def save(self, file_path: str = None, folder_path: str = None, prefix: str = None, features=True, verbose=False):
        if file_path is None:
            if folder_path is None:
                folder_path = self.folder_path
            if prefix is None:
                prefix = self.prefix
            file_path = folder_path + self.name + prefix + '.pth'
        else:
            folder_path = os.path.dirname(file_path)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        _dict = self._model.state_dict() if features else self._model.classifier.state_dict()
        torch.save(_dict, file_path)
        if verbose:
            print('Model {} saved at: '.format(self.name), file_path)

    # define in concrete model class.
    def load_official_weights(self, verbose=True):
        raise NotImplementedError

    #-----------------------------------Train and Validate------------------------------------#
    def _train(self, epoch: int, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler._LRScheduler = None,
               validate_interval=10, save=True, prefix: str = None, verbose=True, indent=0,
               loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
               get_data: function = None, validate_func=None, **kwargs):

        if loader_train is None:
            loader_train = self.dataset.loader['train']
        if verbose:
            loader_train = tqdm(loader_train)
        if get_data is None:
            get_data = self.get_data
        if validate_func is None:
            validate_func = self._validate

        _, best_acc, _ = validate_func(loader=loader_valid, get_data=get_data,
                                       verbose=verbose, indent=indent, **kwargs)
        self.train()

        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        # progress = ProgressMeter(
        #     len(trainloader),
        #     [batch_time, data_time, losses, top1, top5],
        #     prefix="Epoch: [{}]".format(epoch))

        optimizer.zero_grad()
        # start = time.perf_counter()
        # end = start
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            epoch_start = time.perf_counter()
            for data in loader_train:
                # data_time.update(time.perf_counter() - end)
                _input, _label = self.get_data(data, mode='train')
                _output = self.get_logits(_input)
                loss = self.criterion(_output, _label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                losses.update(loss.item(), _label.size(0))
                batch_size = int(_label.size(0))
                top1.update(acc1, batch_size)
                top5.update(acc5, batch_size)

                empty_cache()

                # batch_time.update(time.perf_counter() - end)
                # end = time.perf_counter()

                # if i % 10 == 0:
                #     progress.display(i)
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter()-epoch_start)))
            if verbose:
                pre_str = '{blue_light}Epoch: {0}'.format(
                    output_iter(_epoch+1, epoch), **ansi)
                prints('{:<60}Loss: {:.4f},\tTop1 Acc: {:.3f},\tTop5 Acc: {:.3f}, \t Time: {}'.format(
                    pre_str, losses.avg, top1.avg, top5.avg, epoch_time), prefix='\033[1A\033[K', indent=indent)
            if lr_scheduler:
                lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch+1) % validate_interval == 0 or _epoch == epoch - 1:
                    _, cur_acc, _ = validate_func(loader=loader_valid, get_data=get_data,
                                                  verbose=verbose, indent=indent, **kwargs)
                    self.train()
                    if cur_acc > best_acc and save:
                        self.save(prefix=prefix, verbose=verbose)
                        best_acc = cur_acc
                    if verbose:
                        print('-'*50)
        self.zero_grad()
        self.eval()

    def _validate(self, full=True, loader: torch.utils.data.DataLoader = None, print_prefix='Validate', indent=0, verbose=True, **kwargs):
        self.eval()
        if loader is None:
            loader = self.dataset.loader['valid'] if full else self.dataset.loader['valid2']
        if verbose:
            loader = tqdm(loader)

        # batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        # progress = ProgressMeter(
        #     len(self.dataset.loader['valid']),
        #     [batch_time, losses, top1, top5],
        #     prefix='Test: ')

        # start = time.perf_counter()
        # end = start
        epoch_start = time.perf_counter()
        with torch.no_grad():
            for data in loader:
                _input, _label = self.get_data(data, mode='valid')
                _output = self.get_logits(_input, **kwargs)
                loss = self.criterion(_output, _label)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                losses.update(loss.item(), _label.size(0))

                batch_size = int(_label.size(0))
                top1.update(acc1, batch_size)
                top5.update(acc5, batch_size)

                # empty_cache()

                # measure elapsed time
                # batch_time.update(time.perf_counter() - end)
                # end = time.perf_counter()

                # if i % 10 == 0:
                #     progress.display(i)
        epoch_time = str(datetime.timedelta(seconds=int(
            time.perf_counter()-epoch_start)))
        if verbose:
            pre_str = '{yellow}{0}:{reset}'.format(print_prefix, **ansi)
            prints('{:<35}Loss: {:.4f},\tTop1 Acc: {:.3f},\tTop5 Acc: {:.3f}, \t Time: {}'.format(
                pre_str, losses.avg, top1.avg, top5.avg, epoch_time), prefix='\033[1A\033[K', indent=indent)
        return losses.avg, top1.avg, top5.avg

    #-------------------------------------------Utility---------------------------------------#

    def get_data(self, data, **kwargs):
        if self.dataset is not None:
            return self.dataset.get_data(data, **kwargs)
        else:
            return data

    def accuracy(self, _output: torch.FloatTensor, _label: torch.LongTensor, topk=(1, 5)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = min(max(topk), self.num_classes)
            batch_size = _label.size(0)

            _, pred = _output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(_label.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                if k > self.num_classes:
                    res.append(100.0)
                else:
                    correct_k = correct[:k].view(-1).float().sum(0,
                                                                 keepdim=True)
                    res.append(float(correct_k.mul_(100.0 / batch_size)))
            return res

    def activate_params(self, active_param: list):
        for param in self._model.parameters():
            param.requires_grad = False
        for param in active_param:
            param.requires_grad = True

    def get_parallel(self):
        if env['num_gpus'] > 1:
            if self.dataset is not None:
                if self.dataset.data_type != 'image':
                    return self._model
            elif self.name[0] == 'g' and self.name[2] == 'n':
                return self._model
            return nn.DataParallel(self._model).cuda()
        else:
            return self._model

    @staticmethod
    def output_layer_information(layer, depth=1, indent=0, verbose=False, tree_length=None):
        if tree_length is None:
            tree_length = 10*(depth+1)
        depth -= 1
        if depth >= 0:
            for name, module in layer.named_children():
                _str = '{blue_light}{0}{reset}'.format(name, **ansi)
                if verbose:
                    _str = _str.ljust(
                        tree_length-indent+len(ansi['blue_light'])+len(ansi['reset']))
                    item = str(module).split('\n')[0]
                    if item[-1] == '(':
                        item = item[:-1]
                    _str += item
                prints(_str, indent=indent)
                Model.output_layer_information(
                    module, depth=depth, indent=indent+10, verbose=verbose, tree_length=tree_length)

    def summary(self, indent=0, **kwargs):
        _str = '{blue_light}{0}{reset}'.format(self.name, **ansi)
        prints(_str, indent=indent)
        self.output_layer_information(self._model, indent=indent+10, **kwargs)

    @staticmethod
    def split_name(name, layer=None, default_layer=0, output=False):
        return func(name, layer=layer, default_layer=default_layer, output=output)

        #-----------------------------------------Reload------------------------------------------#
    def __call__(self, *args, **kwargs):
        return self.get_logits(*args, **kwargs)

    # def __str__(self):
    #     return self.summary()

    # def __repr__(self):
    #     return self.summary()

    def train(self, mode=True):
        self._model.train(mode=mode)
        self.model.train(mode=mode)
        return self

    def eval(self):
        self._model.eval()
        self.model.eval()
        return self

    def cuda(self, device=None):
        self._model.cuda(device=device)
        self.model.cuda(device=device)
        return self

    def zero_grad(self):
        self._model.zero_grad()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self._model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self._model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def children(self):
        return self._model.children()

    def named_children(self):
        return self._model.named_children()

    def modules(self):
        return self._model.modules()

    def named_modules(self, memo=None, prefix=''):
        return self._model.named_modules(memo=memo, prefix=prefix)

    def apply(self, fn):
        return self._model.apply(fn)

    #-----------------------------------------------------------------------------------------#

    def remove_misclassify(self, data, **kwargs):
        with torch.no_grad():
            _input, _label = self.get_data(data, **kwargs)
            _classification = self.get_class(_input)
            repeat_idx = _classification.eq(_label)
        return _input[repeat_idx], _label[repeat_idx]

    def generate_target(self, _input: torch.Tensor, idx=1, same=False) -> torch.LongTensor:

        if len(_input.shape) == 3:
            _input = _input.unsqueeze(0)
        self.batch_size = _input.shape[0]
        with torch.no_grad():
            _output = self.get_logits(_input)
        _, indices = _output.sort(dim=-1, descending=True)
        target = torch.as_tensor(
            indices[:, idx], dtype=torch.long, device=_input.device)
        if same:
            target = repeat_to_batch(target.mode(dim=0)[0], len(_input))
        return target
