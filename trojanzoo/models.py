# -*- coding: utf-8 -*-


from trojanzoo.configs import config, Config
from trojanzoo.datasets import Dataset
from trojanzoo.environ import env
from trojanzoo.utils import add_noise, empty_cache, repeat_to_batch, to_tensor
from trojanzoo.utils.output import ansi, prints, output_iter
from trojanzoo.utils import get_name, AverageMeter


import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.cuda.amp
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import argparse
import os
import datetime
import time
from tqdm import tqdm
from collections import OrderedDict
from collections.abc import Callable, Iterable    # TODO: callable (many places) (wait for python update)
from typing import Generator, Iterator, Mapping, Set, Union, Optional

InputType = Union[torch.Tensor, tuple]
# redirect = Indent_Redirect(buffer=True, indent=0)


class _Model(nn.Module):
    def __init__(self, num_classes: int = None, conv_depth: int = 0, conv_dim: int = 0,
                 fc_depth: int = 0, fc_dim: int = 0, **kwargs):
        super().__init__()

        self.conv_depth = conv_depth
        self.conv_dim = conv_dim
        self.fc_depth = fc_depth
        self.fc_dim = fc_dim
        self.num_classes = num_classes

        self.features = self.define_features()   # feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # average pooling
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = self.define_classifier()  # classifier

    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)
    def forward(self, x: InputType) -> torch.Tensor:
        # if x.shape is (channels, height, width)
        # (channels, height, width) ==> (batch_size: 1, channels, height, width)
        x = self.get_final_fm(x)
        x = self.classifier(x)
        return x

    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x: InputType) -> torch.Tensor:
        return self.features(x)

    def get_final_fm(self, x: InputType) -> torch.Tensor:
        x = self.get_fm(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def define_features(self, conv_depth: int = None, conv_dim: int = None) -> nn.Sequential:
        return nn.Sequential(OrderedDict([('id', nn.Identity())]))

    def define_classifier(self, num_classes: int = None, fc_depth: int = None,
                          conv_dim: int = None, fc_dim: int = None,
                          activation: str = 'relu', dropout: bool = True) -> nn.Sequential:
        fc_depth = fc_depth if fc_depth is not None else self.fc_depth
        conv_dim = conv_dim if conv_dim is not None else self.conv_dim
        fc_dim = fc_dim if fc_dim is not None else self.fc_dim
        num_classes = num_classes if num_classes is not None else self.num_classes
        if fc_depth <= 0:
            return nn.Sequential(OrderedDict([('fc', nn.Identity())]))
        dim_list: list[int] = [fc_dim] * (fc_depth - 1)
        dim_list.insert(0, conv_dim)
        ActivationType: type[nn.Module] = nn.ReLU
        if activation == 'sigmoid':
            ActivationType = nn.Sigmoid
        elif not activation == 'relu':
            raise NotImplementedError(f'{activation=}')
        seq = []
        if fc_depth == 1:
            seq.append(('fc', nn.Linear(self.conv_dim, num_classes)))
        else:
            for i in range(fc_depth - 1):
                seq.append(('fc' + str(i + 1), nn.Linear(dim_list[i], dim_list[i + 1])))
                if activation:
                    seq.append((f'{activation}{i + 1:d}', ActivationType(inplace=True)))
                if dropout:
                    seq.append((f'dropout{i + 1:d}', nn.Dropout()))
            seq.append(('fc' + str(self.fc_depth),
                        nn.Linear(self.fc_dim, self.num_classes)))
        return nn.Sequential(OrderedDict(seq))


class Model:

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('-m', '--model', dest='model_name',
                           help='model name, defaults to config[model][default_model]')
        group.add_argument('--layer', dest='layer', type=int,
                           help='layer (optional, maybe embedded in --model)')
        group.add_argument('--suffix', dest='suffix',
                           help='model name suffix, e.g. _adv_train')
        group.add_argument('--pretrain', dest='pretrain', action='store_true',
                           help='load pretrained weights, defaults to False')
        group.add_argument('--official', dest='official', action='store_true',
                           help='load official weights, defaults to False')
        group.add_argument('--model_dir', dest='model_dir',
                           help='directory to contain pretrained models')
        group.add_argument('--randomized_smooth', dest='randomized_smooth', action='store_true',
                           help='whether to use randomized smoothing, defaults to False')
        group.add_argument('--rs_sigma', dest='rs_sigma', type=float,
                           help='randomized smoothing sampling std, defaults to 0.01')
        group.add_argument('--rs_n', dest='rs_n', type=int,
                           help='randomized smoothing sampling number, defaults to 100')
        return group

    def __init__(self, name: str = None, model_class: type[_Model] = _Model, dataset: Dataset = None,
                 num_classes: int = None, folder_path: str = None,
                 official: bool = False, pretrain: bool = False,
                 randomized_smooth: bool = False, rs_sigma: float = 0.01, rs_n: int = 100,
                 suffix: str = '', **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['model'] = ['suffix', 'pretrain', 'official', 'randomized_smooth']
        if randomized_smooth:
            self.param_list['model'].extend(['rs_sigma', 'rs_n'])
        self.name: str = name
        self.dataset = dataset
        self.suffix = suffix
        self.pretrain = pretrain
        self.official = official
        self.randomized_smooth: bool = randomized_smooth
        self.rs_sigma: float = rs_sigma
        self.rs_n: int = rs_n

        self.folder_path = folder_path
        if folder_path is not None:
            self.folder_path = os.path.normpath(folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # ------------Auto-------------- #
        loss_weights: np.ndarray = None if 'loss_weights' not in kwargs.keys() else kwargs['loss_weights']
        if dataset:
            if not isinstance(dataset, Dataset):
                raise TypeError(f'{type(dataset)=}    {dataset=}')
            num_classes = num_classes if num_classes is not None else dataset.num_classes
            loss_weights = loss_weights if 'loss_weights' in kwargs.keys() else dataset.loss_weights
        self.num_classes = num_classes  # number of classes
        self.loss_weights = loss_weights  # TODO: what device shall we save loss_weights? numpy, torch, or torch.cuda.

        # ------------------------------ #
        self.criterion = self.define_criterion(weight=to_tensor(loss_weights))
        self.softmax = nn.Softmax(dim=1)
        self._model = model_class(num_classes=num_classes, **kwargs)
        self.activate_params([])
        if official:
            self.load('official')
        if pretrain:
            self.load()
        self.model = self.get_parallel_model()
        self.eval()
        if env['num_gpus']:  # TODO: might be useless if we set map_location correctly
            self.cuda()

    # ----------------- Forward Operations ----------------------#

    def get_logits(self, _input: InputType, randomized_smooth: bool = None,
                   rs_sigma: float = None, rs_n: int = None, **kwargs) -> torch.Tensor:
        randomized_smooth = randomized_smooth if randomized_smooth is not None else self.randomized_smooth
        if randomized_smooth:
            rs_sigma = rs_sigma if rs_sigma is not None else self.rs_sigma
            rs_n = rs_n if rs_n is not None else self.rs_n
            _list = []
            for _ in range(rs_n):
                _input_noise = add_noise(_input, std=rs_sigma)
                _list.append(self.model(_input_noise, **kwargs))
            return torch.stack(_list).mean(dim=0)
            # TODO: memory issues and parallel possibilities
            # _input_noise = add_noise(repeat_to_batch(_input, batch_size=n), std=sigma).flatten(end_dim=1)
            # return self.model(_input_noise, **kwargs).view(n, len(_input), self.num_classes).mean(dim=0)
        else:
            return self.model(_input, **kwargs)

    def get_prob(self, _input: InputType, **kwargs) -> torch.Tensor:
        return self.softmax(self.get_logits(_input, **kwargs))

    def get_final_fm(self, _input: InputType, **kwargs) -> torch.Tensor:
        return self._model.get_final_fm(_input, **kwargs)

    def get_target_prob(self, _input: InputType, target: Union[torch.Tensor, list[int]],
                        **kwargs) -> torch.Tensor:
        if isinstance(target, list):
            target = torch.tensor(target, device=_input.device)
        return self.get_prob(_input, **kwargs).gather(dim=1, index=target.unsqueeze(1)).flatten()

    def get_class(self, _input: InputType, **kwargs) -> torch.Tensor:
        return self.get_logits(_input, **kwargs).argmax(dim=-1)

    def loss(self, _input: InputType, _label: torch.Tensor, **kwargs) -> torch.Tensor:
        _output = self(_input, **kwargs)
        return self.criterion(_output, _label)

    # -------------------------------------------------------- #

    # Define the optimizer
    # and transfer to that tuning mode.
    # train_opt: 'full' or 'partial' (default: 'partial')
    # lr: (default: [full:2e-3, partial:2e-4])
    # OptimType: to be implemented
    #
    # return: optimizer

    def define_optimizer(self, parameters: Union[str, Iterator[nn.Parameter]] = 'full',
                         OptimType: Union[str, type[Optimizer]] = None,
                         lr_scheduler: bool = True,
                         lr: float = 0.1, lr_decay_step: int = 30,
                         **kwargs) -> tuple[Optimizer, _LRScheduler]:
        if isinstance(parameters, str):
            parameters = self.get_parameter_from_name(name=parameters)
        if not isinstance(parameters, Iterable):
            raise TypeError(f'{type(parameters)=}    {parameters=}')
        if isinstance(OptimType, str):
            OptimType: type[Optimizer] = getattr(torch.optim, OptimType)
        if len(kwargs) == 0 and OptimType == torch.optim.SGD:
            kwargs = {'momentum': 0.9, 'weight_decay': 2e-4, 'nesterov': True}
        optimizer = OptimType(parameters, lr, **kwargs)
        _lr_scheduler: _LRScheduler = None
        if lr_scheduler:
            _lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_decay_step, gamma=0.1)
            # optimizer = optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[150, 250], gamma=0.1)
        return optimizer, _lr_scheduler

    # define loss function
    # Cross Entropy
    def define_criterion(self, **kwargs) -> nn.CrossEntropyLoss:    # TODO: linting, or maybe nn.Module for generic?
        if 'weight' not in kwargs.keys():
            kwargs['weight'] = self.loss_weights
        return nn.CrossEntropyLoss(**kwargs)

    # -----------------------------Load & Save Model------------------------------------------- #

    # file_path: (default: '') if '', use the default path. Else if the path doesn't exist, quit.
    # full: (default: False) whether save feature extractor.
    # output: (default: False) whether output help information.
    def load(self, file_path: str = None, folder_path: str = None, suffix: str = None,
             map_location: Union[str, Callable, torch.device, dict] = 'default',
             component: str = '', strict: bool = True,
             verbose: bool = False, indent: int = 0, **kwargs):
        map_location = map_location if map_location != 'default' else env['device']
        if file_path is None:
            folder_path = folder_path if folder_path is not None else self.folder_path
            suffix = suffix if suffix is not None else self.suffix
            file_path = os.path.normpath(os.path.join(folder_path, f'{self.name}{suffix}.pth'))
        if file_path == 'official':   # TODO
            _dict = self.get_official_weights(map_location=map_location)
            last_bias_value = next(reversed(_dict.values()))   # TODO: make sure
            if self.num_classes != len(last_bias_value) and component != 'features':
                strict = False
                _dict.popitem()
                _dict.popitem()
        else:
            try:
                # TODO: type annotation might change? dict[str, torch.Tensor]
                _dict: OrderedDict[str, torch.Tensor] = torch.load(file_path, map_location=map_location, **kwargs)
            except Exception as e:
                print(f'{file_path=}')
                raise e
        module = self._model
        if component == 'features':
            module = self._model.features
            _dict = OrderedDict([(key.removeprefix('features.'), value) for key, value in _dict.items()])
        elif component == 'classifier':
            module = self._model.classifier
            _dict = OrderedDict([(key.removeprefix('classifier.'), value) for key, value in _dict.items()])
        else:
            assert component == '', f'{component=}'
        module.load_state_dict(_dict, strict=strict)
        if verbose:
            prints(f'Model {self.name} loaded from: {file_path}', indent=indent)

    # file_path: (default: '') if '', use the default path.
    # full: (default: False) whether save feature extractor.
    def save(self, file_path: str = None, folder_path: str = None, suffix: str = None,
             component: str = '', verbose: bool = False, indent: int = 0, **kwargs):
        if file_path is None:
            folder_path = folder_path if folder_path is not None else self.folder_path
            suffix = suffix if suffix is not None else self.suffix
            file_path = os.path.normpath(os.path.join(folder_path, f'{self.name}{suffix}.pth'))
        else:
            folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # TODO: type annotation might change? dict[str, torch.Tensor]
        module = self._model
        if component == 'features':
            module = self._model.features
        elif component == 'classifier':
            module = self._model.classifier
        else:
            assert component == '', f'{component=}'
        _dict: OrderedDict[str, torch.Tensor] = module.state_dict(prefix=component)
        torch.save(_dict, file_path, **kwargs)
        if verbose:
            prints(f'Model {self.name} saved at: {file_path}', indent=indent)

    # define in concrete model class.
    # TODO: maybe write some generic style? model_url?
    def get_official_weights(self, map_location: Union[str, Callable, torch.device, dict] = 'default',
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        raise NotImplementedError(f'{self.name} has no official weights.')

    # -----------------------------------Train and Validate------------------------------------ #
    def _train(self, epoch: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
               validate_interval: int = 10, save: bool = False, amp: bool = False, verbose: bool = True, indent: int = 0,
               loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
               get_data_fn: Callable[..., tuple[InputType, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               validate_func: Callable[..., tuple[float, ...]] = None, epoch_func: Callable[[], None] = None,
               save_fn: Callable = None, file_path: str = None, folder_path: str = None, suffix: str = None, **kwargs):
        loader_train = loader_train if loader_train is not None else self.dataset.loader['train']
        get_data_fn = get_data_fn if get_data_fn is not None else self.get_data
        loss_fn = loss_fn if loss_fn is not None else self.loss
        validate_func = validate_func if validate_func is not None else self._validate
        save_fn = save_fn if save_fn is not None else self.save

        scaler: torch.cuda.amp.GradScaler = None
        if amp and env['num_gpus']:
            scaler = torch.cuda.amp.GradScaler()
        _, best_acc, _ = validate_func(loader=loader_valid, get_data_fn=get_data_fn, loss_fn=loss_fn,
                                       verbose=verbose, indent=indent, **kwargs)
        losses = AverageMeter('Loss')
        top1 = AverageMeter('Acc@1')
        top5 = AverageMeter('Acc@5')
        params: list[list[nn.Parameter]] = [param_group['params'] for param_group in optimizer.param_groups]
        for _epoch in range(epoch):
            if epoch_func is not None:
                self.activate_params([])
                epoch_func()
                self.activate_params(params)
            losses.reset()
            top1.reset()
            top5.reset()
            epoch_start = time.perf_counter()
            loader = loader_train
            if verbose and env['tqdm']:
                loader = tqdm(loader_train)
            self.train()
            self.activate_params(params)
            optimizer.zero_grad()
            for data in loader:
                # data_time.update(time.perf_counter() - end)
                _input, _label = get_data_fn(data, mode='train')
                if amp and env['num_gpus']:
                    loss = loss_fn(_input, _label, amp=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = loss_fn(_input, _label)
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    _output = self.get_logits(_input)
                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                batch_size = int(_label.size(0))
                losses.update(loss.item(), batch_size)
                top1.update(acc1, batch_size)
                top5.update(acc5, batch_size)
                empty_cache()
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            self.eval()
            self.activate_params([])
            if verbose:
                pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                    output_iter(_epoch + 1, epoch), **ansi).ljust(64 if env['color'] else 35)
                _str = ' '.join([
                    f'Loss: {losses.avg:.4f},'.ljust(20),
                    f'Top1 Acc: {top1.avg:.3f}, '.ljust(20),
                    f'Top5 Acc: {top5.avg:.3f},'.ljust(20),
                    f'Time: {epoch_time},'.ljust(20),
                ])
                prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi) if env['tqdm'] else '',
                       indent=indent)
            if lr_scheduler:
                lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch + 1) % validate_interval == 0 or _epoch == epoch - 1:
                    _, cur_acc, _ = validate_func(loader=loader_valid, get_data_fn=get_data_fn, loss_fn=loss_fn,
                                                  verbose=verbose, indent=indent, **kwargs)
                    if cur_acc >= best_acc:
                        prints('best result update!', indent=indent)
                        prints(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}', indent=indent)
                        best_acc = cur_acc
                        if save:
                            save_fn(file_path=file_path, folder_path=folder_path, suffix=suffix, verbose=verbose)
                    if verbose:
                        print('-' * 50)
        self.zero_grad()

    def _validate(self, full=True, print_prefix='Validate', indent=0, verbose=True,
                  loader: torch.utils.data.DataLoader = None,
                  get_data_fn: Callable = None, loss_fn: Callable[..., float] = None, **kwargs) -> tuple[float, ...]:
        self.eval()
        if loader is None:
            loader = self.dataset.loader['valid'] if full else self.dataset.loader['valid2']
        get_data_fn = get_data_fn if get_data_fn is not None else self.get_data
        loss_fn = loss_fn if loss_fn is not None else self.loss
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        epoch_start = time.perf_counter()
        if verbose and env['tqdm']:
            loader = tqdm(loader)
        for data in loader:
            _input, _label = get_data_fn(data, mode='valid', **kwargs)
            with torch.no_grad():
                loss = loss_fn(_input, _label)
                _output = self.get_logits(_input)
            # measure accuracy and record loss
            batch_size = int(_label.size(0))
            losses.update(loss.item(), _label.size(0))
            acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
        epoch_time = str(datetime.timedelta(seconds=int(
            time.perf_counter() - epoch_start)))
        if verbose:
            pre_str = '{yellow}{0}:{reset}'.format(print_prefix, **ansi).ljust(35)
            _str = ' '.join([
                f'Loss: {losses.avg:.4f},'.ljust(20),
                f'Top1 Acc: {top1.avg:.3f}, '.ljust(20),
                f'Top5 Acc: {top5.avg:.3f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi) if env['tqdm'] else '', indent=indent)
        return losses.avg, top1.avg, top5.avg

    # -------------------------------------------Utility--------------------------------------- #

    def get_data(self, data: tuple[InputType, torch.Tensor], **kwargs):
        if self.dataset is not None:
            return self.dataset.get_data(data, **kwargs)
        else:
            return data

    def accuracy(self, _output: torch.Tensor, _label: torch.Tensor,
                 topk: tuple[int] = (1, 5)) -> tuple[float, ...]:
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = min(max(topk), self.num_classes)
            batch_size = _label.shape[0]
            _, pred = _output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(_label.view(1, -1).expand_as(pred))
            res: tuple[float, ...] = []
            for k in topk:
                if k > self.num_classes:
                    res.append(100.0)
                else:
                    correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
                    res.append(float(correct_k.mul_(100.0 / batch_size)))
            return res

    def get_parameter_from_name(self, name: str = '') -> Iterator[nn.Parameter]:
        params = self._model.parameters()
        if name == 'features':
            params = self._model.features.parameters()
        elif name in ['classifier', 'partial']:
            params = self._model.classifier.parameters()
        elif name not in ['', 'full']:
            raise NotImplementedError(f'{name=}')
        return params

    def activate_params(self, param_groups: list[list[nn.Parameter]]):
        for param in self._model.parameters():
            param.requires_grad_(False)
        for param_group in param_groups:
            for param in param_group:
                param.requires_grad_()

    # Need to overload for other packages (GNN) since they are calling their own nn.DataParallel.
    def get_parallel_model(self) -> Union[_Model, nn.DataParallel]:
        if env['num_gpus'] > 1:
            return nn.DataParallel(self._model)
        return self._model

    @staticmethod
    def output_layer_information(layer: nn.Module, depth: int = 0, verbose: bool = True,
                                 indent: int = 0, tree_length: int = None):
        tree_length = tree_length if tree_length is not None else 10 * (depth + 1)
        if depth > 0:
            for name, module in layer.named_children():
                _str = '{blue_light}{0}{reset}'.format(name, **ansi)
                if verbose:
                    _str = _str.ljust(
                        tree_length - indent + len(ansi['blue_light']) + len(ansi['reset']))
                    item = str(module).split('\n')[0]
                    if item[-1] == '(':
                        item = item[:-1]
                    _str += item
                prints(_str, indent=indent)
                Model.output_layer_information(
                    module, depth=depth - 1, indent=indent + 10, verbose=verbose, tree_length=tree_length)

    def summary(self, depth: int = 2, verbose: bool = True, indent: int = 0, **kwargs):
        prints('{blue_light}{0:<20s}{reset} Parameters: '.format(self.name, **ansi), indent=indent)
        for key, value in self.param_list.items():
            prints('{green}{0:<20s}{reset}'.format(key, **ansi), indent=indent + 10)
            prints({v: getattr(self, v) for v in value}, indent=indent + 10)
            prints('-' * 20, indent=indent + 10)
        self.output_layer_information(self._model, depth=depth, verbose=verbose, indent=indent + 10, **kwargs)
        prints('-' * 20, indent=indent + 10)

    # -----------------------------------------Reload------------------------------------------ #

    def __call__(self, _input: InputType, amp: bool = False, **kwargs) -> torch.Tensor:
        if amp:
            with torch.cuda.amp.autocast():
                return self.get_logits(_input, **kwargs)
        return self.get_logits(_input, **kwargs)

    # def __str__(self) -> str:
    #     sys.stdout = redirect
    #     self.summary()
    #     _str = redirect.buffer
    #     redirect.reset()
    #     return _str

    # def __repr__(self):
    #     return self.name

    def train(self, mode: bool = True):
        self._model.train(mode=mode)
        self.model.train(mode=mode)
        return self

    def eval(self):
        self._model.eval()
        self.model.eval()
        return self

    def cpu(self):
        self._model.cpu()
        self.model.cpu()
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        self._model.cuda(device=device)
        self.model.cuda(device=device)
        return self

    def zero_grad(self, set_to_none: bool = False):
        return self._model.zero_grad(set_to_none=set_to_none)

    def state_dict(self, destination: Mapping[str, torch.Tensor] = None, prefix: str = '', keep_vars: bool = False):
        return self._model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        return self._model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse: bool = True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def children(self):
        return self._model.children()

    def named_children(self):
        return self._model.named_children()

    def modules(self):
        return self._model.modules()

    # TODO: why use the '' for 'nn.Module'?
    def named_modules(self, memo: Optional[Set['nn.Module']] = None, prefix: str = '') -> Generator[tuple[str, nn.Module], None, None]:
        return self._model.named_modules(memo=memo, prefix=prefix)

    def apply(self, fn: Callable[['nn.Module'], None]):
        return self._model.apply(fn)

    # ----------------------------------------------------------------------------------------- #

    def remove_misclassify(self, data: tuple[InputType, torch.Tensor], **kwargs):
        with torch.no_grad():
            _input, _label = self.get_data(data, **kwargs)
            _classification = self.get_class(_input)
            repeat_idx = _classification.eq(_label)
        return _input[repeat_idx], _label[repeat_idx]

    def generate_target(self, _input: InputType, idx: int = 1, same: bool = False) -> torch.Tensor:
        with torch.no_grad():
            _output = self.get_logits(_input)
        target = _output.argsort(dim=-1, descending=True)[:, idx]
        if same:
            target = repeat_to_batch(target.mode(dim=0)[0], len(_input))
        return target


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: Union[str, Model] = None,
                 config: Config = config, class_dict: dict[str, type[Model]] = None) -> argparse._ArgumentGroup:
    dataset_name = get_name(arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']

    group = parser.add_argument_group('{yellow}model{reset}'.format(**ansi), description=model_name)
    ModelType = class_dict[model_name]
    return ModelType.add_argument(group)     # TODO: Linting problem


def create(model_name: str = None, model: Union[str, Model] = None, folder_path: str = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           config: Config = config, class_dict: dict[str, type[Model]] = {}, **kwargs) -> Model:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    result = config.get_config(dataset_name=dataset_name)['model']._update(kwargs)
    model_name = model_name if model_name is not None else result['default_model']

    ModelType: type[Model] = class_dict[model_name]
    if folder_path is None and isinstance(dataset, Dataset):
        folder_path = os.path.join(result['model_dir'], dataset.data_type, dataset.name)
    return ModelType(name=model_name, dataset=dataset, folder_path=folder_path, **result)
