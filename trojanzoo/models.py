#!/usr/bin/env python3

from trojanzoo.configs import config
from trojanzoo.datasets import Dataset
from trojanzoo.environ import env
from trojanzoo.utils import add_noise, empty_cache, repeat_to_batch, to_tensor
from trojanzoo.utils import get_name
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.model import get_all_layer, get_layer, get_layer_name
from trojanzoo.utils.output import ansi, get_ansi_len, prints, output_iter

import torch
import torch.nn as nn
from torch.utils import model_zoo
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict
from collections.abc import Iterable    # TODO: callable (many places) (wait for python update)

from typing import TYPE_CHECKING
from typing import Generator, Iterator, Mapping, Optional, Set, Union    # TODO: python 3.10
from trojanzoo.configs import Config    # TODO: python 3.10
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import argparse
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data
# redirect = Indent_Redirect(buffer=True, indent=0)

__all__ = ['Model', 'add_argument', 'create',
           'get_available_models', 'output_available_models', 'get_model_class']


class _Model(nn.Module):
    def __init__(self, num_classes: int = None, **kwargs):
        super().__init__()
        self.define_preprocess(**kwargs)
        self.features = self.define_features(**kwargs)   # feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # average pooling
        self.flatten = nn.Flatten()
        self.classifier = self.define_classifier(num_classes=num_classes, **kwargs)  # classifier
        self.softmax = nn.Softmax(dim=1)

        self.num_classes = num_classes

    def define_preprocess(self, **kwargs):
        pass

    @staticmethod
    def define_features(**kwargs) -> nn.Module:
        return nn.Identity()

    @staticmethod
    def define_classifier(conv_dim: int = 0, num_classes: int = None,
                          fc_depth: int = 0, fc_dim: int = 0,
                          activation: str = 'relu', dropout: bool = True,
                          **kwargs) -> nn.Sequential:
        seq = nn.Sequential()
        if fc_depth <= 0:
            return seq
        dim_list: list[int] = [fc_dim] * (fc_depth - 1)
        dim_list.insert(0, conv_dim)
        ActivationType: type[nn.Module] = nn.ReLU
        if activation == 'sigmoid':
            ActivationType = nn.Sigmoid
        elif not activation == 'relu':
            raise NotImplementedError(f'{activation=}')
        if fc_depth == 1:
            seq.add_module('fc', nn.Linear(conv_dim, num_classes))
        else:
            for i in range(fc_depth - 1):
                seq.add_module(f'fc{i + 1:d}', nn.Linear(dim_list[i], dim_list[i + 1]))
                if activation:
                    seq.add_module(f'{activation}{i + 1:d}', ActivationType(True))
                if dropout:
                    seq.add_module(f'dropout{i + 1:d}', nn.Dropout())
            seq.add_module(f'fc{fc_depth:d}', nn.Linear(fc_dim, num_classes))
        return seq

    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.get_final_fm(x, **kwargs)
        x = self.classifier(x)
        return x

    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.features(x)

    def get_final_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.get_fm(x, **kwargs)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class Model:
    available_models: list[str] = []
    model_urls: dict[str, str] = []

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('-m', '--model', dest='model_name',
                           help='model name, defaults to config[model][default_model]')
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

    def __init__(self, name: str = None, model: Union[type[_Model], _Model] = _Model,
                 dataset: Dataset = None,
                 num_classes: int = None, folder_path: str = None,
                 official: bool = False, pretrain: bool = False,
                 randomized_smooth: bool = False, rs_sigma: float = 0.01, rs_n: int = 100,
                 suffix: str = '', **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['model'] = ['folder_path']
        if suffix:
            self.param_list['model'].append('suffix')
        if randomized_smooth:
            self.param_list['model'].extend(['randomized_smooth', 'rs_sigma', 'rs_n'])
        self.name: str = name
        self.dataset = dataset
        self.suffix = suffix
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
        self.layer_name_list: list[str] = None

        # ------------------------------ #
        self.criterion = self.define_criterion(weight=to_tensor(loss_weights))
        if isinstance(model, type):
            if num_classes is not None:
                kwargs['num_classes'] = num_classes
            self._model = model(name=name, **kwargs)
        else:
            assert isinstance(model, nn.Module)
            self._model = model
        self.model = self.get_parallel_model(self._model)
        self.activate_params([])
        if official:
            self.load('official')
        if pretrain:
            self.load()
        self.eval()
        if env['num_gpus']:
            self.cuda()

    # ----------------- Forward Operations ----------------------#

    def get_logits(self, _input: torch.Tensor, randomized_smooth: bool = None,
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

    def get_final_fm(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model.get_final_fm(_input, **kwargs)

    def get_prob(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model.softmax(self(_input, **kwargs))

    def get_target_prob(self, _input: torch.Tensor, target: Union[torch.Tensor, list[int]],
                        **kwargs) -> torch.Tensor:
        if isinstance(target, list):
            target = torch.tensor(target, device=_input.device)
        return self.get_prob(_input, **kwargs).gather(dim=1, index=target.unsqueeze(1)).flatten()

    def get_class(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(_input, **kwargs).argmax(dim=-1)

    def get_layer_name(self, depth: int = -1, prefix: str = '',
                       use_filter: bool = True, repeat: bool = False,
                       seq_only: bool = False) -> list[str]:
        return get_layer_name(self._model, depth, prefix, use_filter, repeat, seq_only)

    def get_all_layer(self, x: torch.Tensor,
                      layer_input: str = 'input', depth: int = 0,
                      prefix='', use_filter: bool = True, repeat: bool = False,
                      seq_only: bool = True) -> dict[str, torch.Tensor]:
        return get_all_layer(self._model, x, layer_input, depth,
                             prefix, use_filter, repeat, seq_only)

    def get_layer(self, x: torch.Tensor, layer_output: str = 'classifier',
                  layer_input: str = 'input', prefix: str = '', seq_only: bool = True) -> torch.Tensor:
        if layer_input == 'input':
            if layer_output == 'classifier':
                return self(x)
            elif layer_output == 'features':
                return self._model.get_fm(x)
            elif layer_output == 'flatten':
                return self.get_final_fm(x)
        if self.layer_name_list is None:
            self.layer_name_list: list[str] = self.get_layer_name()
            self.layer_name_list.insert(0, 'input')
            self.layer_name_list.append('output')
        return get_layer(self._model, x, layer_output, layer_input, prefix,
                         layer_name_list=self.layer_name_list, seq_only=seq_only)

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if _output is None:
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
                         lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0,
                         lr_scheduler: bool = True, T_max: int = None,
                         **kwargs) -> tuple[Optimizer, _LRScheduler]:
        kwargs['momentum'] = momentum
        kwargs['weight_decay'] = weight_decay
        if isinstance(parameters, str):
            parameters = self.get_parameter_from_name(name=parameters)
        if not isinstance(parameters, Iterable):
            raise TypeError(f'{type(parameters)=}    {parameters=}')
        if isinstance(OptimType, str):
            OptimType: type[Optimizer] = getattr(torch.optim, OptimType)
        keys = OptimType.__init__.__code__.co_varnames
        kwargs = {k: v for k, v in kwargs.items() if k in keys}
        optimizer = OptimType(parameters, lr, **kwargs)
        _lr_scheduler: _LRScheduler = None
        if lr_scheduler:
            _lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return optimizer, _lr_scheduler

    # define loss function
    # Cross Entropy
    # TODO: linting, or maybe nn.Module for generic?
    def define_criterion(self, **kwargs) -> nn.CrossEntropyLoss:
        if 'weight' not in kwargs.keys():
            kwargs['weight'] = self.loss_weights
        return nn.CrossEntropyLoss(**kwargs)
        # if loss_type == 'jsd':
        #     num_classes = num_classes if num_classes is not None else self.num_classes

        #     def jsd(_output: torch.Tensor, _label: torch.Tensor, **kwargs):
        #         p = onehot_label(_label, num_classes)
        #         q: torch.Tensor = F.softmax(_output)
        #         log_q = F.log_softmax(_output)
        #         sum_pq = p + q
        #         loss = sum_pq * (sum_pq.log() - math.log(2))
        # return criterion

    # -----------------------------Load & Save Model------------------------------------------- #

    # file_path: (default: '') if '', use the default path. Else if the path doesn't exist, quit.
    # full: (default: False) whether save feature extractor.
    # output: (default: False) whether output help information.
    def load(self, file_path: str = None, folder_path: str = None, suffix: str = None,
             map_location: Union[str, Callable, torch.device, dict] = 'cpu',
             component: str = '', strict: bool = True,
             verbose: bool = False, indent: int = 0, **kwargs):
        with torch.no_grad():
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
            try:
                module.load_state_dict(_dict, strict=strict)
            except RuntimeError as e:
                prints(f'Model {self.name} loaded from: {file_path}', indent=indent)
                raise e
            if verbose:
                prints(f'Model {self.name} loaded from: {file_path}', indent=indent)
            if env['num_gpus']:
                self.cuda()

    # file_path: (default: '') if '', use the default path.
    # full: (default: False) whether save feature extractor.
    def save(self, file_path: str = None, folder_path: str = None, suffix: str = None,
             component: str = '', verbose: bool = False, indent: int = 0, **kwargs):
        with torch.no_grad():
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

    def get_official_weights(self, url: str = None,
                             map_location: Union[str, Callable, torch.device, dict] = 'cpu',
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = self.model_urls[self.name] if url is None else url
        print('get official model weights from: ', url)
        return model_zoo.load_url(url, map_location=map_location, **kwargs)

    # -----------------------------------Train and Validate------------------------------------ #
    def _train(self, epoch: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None, grad_clip: float = None,
               print_prefix: str = 'Epoch', start_epoch: int = 0,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
               epoch_fn: Callable[..., None] = None,
               get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               after_loss_fn: Callable[..., None] = None,
               validate_fn: Callable[..., tuple[float, float]] = None,
               save_fn: Callable[..., None] = None, file_path: str = None, folder_path: str = None, suffix: str = None,
               writer=None, main_tag: str = 'train', tag: str = '',
               verbose: bool = True, indent: int = 0, **kwargs):
        loader_train = loader_train if loader_train is not None else self.dataset.loader['train']
        get_data_fn = get_data_fn if callable(get_data_fn) else self.get_data
        loss_fn = loss_fn if callable(loss_fn) else self.loss
        validate_fn = validate_fn if callable(validate_fn) else self._validate
        save_fn = save_fn if callable(save_fn) else self.save
        # if not callable(iter_fn) and hasattr(self, 'iter_fn'):
        #     iter_fn = getattr(self, 'iter_fn')
        if not callable(epoch_fn) and hasattr(self, 'epoch_fn'):
            epoch_fn = getattr(self, 'epoch_fn')
        if not callable(after_loss_fn) and hasattr(self, 'after_loss_fn'):
            after_loss_fn = getattr(self, 'after_loss_fn')

        scaler: torch.cuda.amp.GradScaler = None
        if not env['num_gpus']:
            amp = False
        if amp:
            scaler = torch.cuda.amp.GradScaler()
        _, best_acc = validate_fn(loader=loader_valid, get_data_fn=get_data_fn, loss_fn=loss_fn,
                                  writer=None, tag=tag, _epoch=start_epoch,
                                  verbose=verbose, indent=indent, **kwargs)

        params: list[nn.Parameter] = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        total_iter = epoch * len(loader_train)
        for _epoch in range(epoch):
            _epoch += 1
            if callable(epoch_fn):
                self.activate_params([])
                epoch_fn(optimizer=optimizer, lr_scheduler=lr_scheduler,
                         _epoch=_epoch, epoch=epoch, start_epoch=start_epoch)
                self.activate_params(params)
            logger = MetricLogger()
            logger.meters['loss'] = SmoothedValue()
            logger.meters['top1'] = SmoothedValue()
            logger.meters['top5'] = SmoothedValue()
            loader_epoch = loader_train
            if verbose:
                header = '{blue_light}{0}: {1}{reset}'.format(
                    print_prefix, output_iter(_epoch, epoch), **ansi)
                header = header.ljust(30 + get_ansi_len(header))
                if env['tqdm']:
                    header = '{upline}{clear_line}'.format(**ansi) + header
                    loader_epoch = tqdm(loader_epoch)
                loader_epoch = logger.log_every(loader_epoch, header=header, indent=indent)
            self.train()
            self.activate_params(params)
            optimizer.zero_grad()
            for i, data in enumerate(loader_epoch):
                _iter = _epoch * len(loader_train) + i
                # data_time.update(time.perf_counter() - end)
                _input, _label = get_data_fn(data, mode='train')
                _output = self(_input, amp=amp)
                loss = loss_fn(_input, _label, _output=_output, amp=amp)
                if amp:
                    scaler.scale(loss).backward()
                    if callable(after_loss_fn):
                        after_loss_fn(_input=_input, _label=_label, _output=_output,
                                      loss=loss, optimizer=optimizer, loss_fn=loss_fn,
                                      amp=amp, scaler=scaler,
                                      _iter=_iter, total_iter=total_iter)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(params)
                    if callable(after_loss_fn):
                        after_loss_fn(_input=_input, _label=_label, _output=_output,
                                      loss=loss, optimizer=optimizer, loss_fn=loss_fn,
                                      amp=amp, scaler=scaler,
                                      _iter=_iter, total_iter=total_iter)
                        # start_epoch=start_epoch, _epoch=_epoch, epoch=epoch)
                    optimizer.step()
                optimizer.zero_grad()
                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                batch_size = int(_label.size(0))
                logger.meters['loss'].update(float(loss), batch_size)
                logger.meters['top1'].update(acc1, batch_size)
                logger.meters['top5'].update(acc5, batch_size)
                empty_cache()   # TODO: should it be outside of the dataloader loop?
            self.eval()
            self.activate_params([])
            loss, acc = logger.meters['loss'].global_avg, logger.meters['top1'].global_avg
            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                writer.add_scalars(main_tag='Loss/' + main_tag, tag_scalar_dict={tag: loss},
                                   global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='Acc/' + main_tag, tag_scalar_dict={tag: acc},
                                   global_step=_epoch + start_epoch)
            if lr_scheduler:
                lr_scheduler.step()
            if validate_interval != 0:
                if _epoch % validate_interval == 0 or _epoch == epoch:
                    _, cur_acc = validate_fn(loader=loader_valid, get_data_fn=get_data_fn, loss_fn=loss_fn,
                                             writer=writer, tag=tag, _epoch=_epoch + start_epoch,
                                             verbose=verbose, indent=indent, **kwargs)
                    if cur_acc >= best_acc:
                        if verbose:
                            prints('{green}best result update!{reset}'.format(**ansi), indent=indent)
                            prints(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}', indent=indent)
                        best_acc = cur_acc
                        if save:
                            save_fn(file_path=file_path, folder_path=folder_path, suffix=suffix, verbose=verbose)
                    if verbose:
                        prints('-' * 50, indent=indent)
        self.zero_grad()

    def _validate(self, full=True, print_prefix='Validate', indent=0, verbose=True,
                  loader: torch.utils.data.DataLoader = None,
                  get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                  loss_fn: Callable[..., torch.Tensor] = None,
                  writer=None, main_tag: str = 'valid', tag: str = '', _epoch: int = None,
                  **kwargs) -> tuple[float, float]:
        self.eval()
        if loader is None:
            loader = self.dataset.loader['valid'] if full else self.dataset.loader['valid2']
        get_data_fn = get_data_fn if get_data_fn is not None else self.get_data
        loss_fn = loss_fn if loss_fn is not None else self.loss
        logger = MetricLogger()
        logger.meters['loss'] = SmoothedValue()
        logger.meters['top1'] = SmoothedValue()
        logger.meters['top5'] = SmoothedValue()
        loader_epoch = loader
        if verbose:
            header = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
            header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
            if env['tqdm']:
                header = '{upline}{clear_line}'.format(**ansi) + header
                loader_epoch = tqdm(loader_epoch)
            loader_epoch = logger.log_every(loader_epoch, header=header, indent=indent)
        for data in loader_epoch:
            _input, _label = get_data_fn(data, mode='valid', **kwargs)
            with torch.no_grad():
                _output = self(_input)
                loss = float(loss_fn(_input, _label, _output=_output, **kwargs))
                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                batch_size = int(_label.size(0))
                logger.meters['loss'].update(loss, batch_size)
                logger.meters['top1'].update(acc1, batch_size)
                logger.meters['top5'].update(acc5, batch_size)
        loss, acc = logger.meters['loss'].global_avg, logger.meters['top1'].global_avg
        if writer is not None and _epoch is not None and main_tag:
            from torch.utils.tensorboard import SummaryWriter
            assert isinstance(writer, SummaryWriter)
            writer.add_scalars(main_tag='Loss/' + main_tag, tag_scalar_dict={tag: loss}, global_step=_epoch)
            writer.add_scalars(main_tag='Acc/' + main_tag, tag_scalar_dict={tag: acc}, global_step=_epoch)
        return loss, acc

    # -------------------------------------------Utility--------------------------------------- #

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dataset is not None:
            return self.dataset.get_data(data, **kwargs)
        else:
            return data

    def accuracy(self, _output: torch.Tensor, _label: torch.Tensor,
                 topk: tuple[int] = (1, 5)) -> tuple[float, ...]:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = min(max(topk), self.num_classes)
            batch_size = _label.size(0)
            _, pred = _output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(_label[None])
            res: list[float] = []
            for k in topk:
                if k > self.num_classes:
                    res.append(100.0)
                else:
                    correct_k = float(correct[:k].sum(dtype=torch.float32))
                    res.append(correct_k * (100.0 / batch_size))
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

    def activate_params(self, params: Iterator[nn.Parameter]):
        for param in self._model.parameters():
            param.requires_grad_(False)
        for param in params:
            param.requires_grad_()

    # Need to overload for other packages (GNN) since they are calling their own nn.DataParallel.
    # TODO: nn.parallel.DistributedDataParallel
    @staticmethod
    def get_parallel_model(_model: _Model) -> Union[_Model, nn.DataParallel]:
        if env['num_gpus'] > 1:
            return nn.DataParallel(_model)
        return _model

    @staticmethod
    def output_layer_information(layer: nn.Module, depth: int = 0, verbose: bool = True,
                                 indent: int = 0, tree_length: int = None):
        tree_length = tree_length if tree_length is not None else 12 * (depth + 1)
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
                    module, depth=depth - 1, indent=indent + 12, verbose=verbose, tree_length=tree_length)

    def summary(self, depth: int = None, verbose: bool = True, indent: int = 0, **kwargs):
        if depth is None:
            depth = env['verbose']
        if depth is None:
            depth = 1
        prints('{blue_light}{0:<30s}{reset} Parameters: '.format(self.name, **ansi), indent=indent)
        prints(self.__class__.__name__, indent=indent)
        for key, value in self.param_list.items():
            prints('{green}{0:<20s}{reset}'.format(key, **ansi), indent=indent + 10)
            prints({v: getattr(self, v) for v in value}, indent=indent + 10)
            prints('-' * 20, indent=indent + 10)
        self.output_layer_information(self._model, depth=depth, verbose=verbose, indent=indent + 10, **kwargs)
        prints('-' * 20, indent=indent + 10)

    # -----------------------------------------Reload------------------------------------------ #

    def __call__(self, _input: torch.Tensor, amp: bool = False, **kwargs) -> torch.Tensor:
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

    def remove_misclassify(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs):
        with torch.no_grad():
            _input, _label = self.get_data(data, **kwargs)
            _classification = self.get_class(_input)
            repeat_idx = _classification.eq(_label)
        return _input[repeat_idx], _label[repeat_idx]

    def generate_target(self, _input: torch.Tensor, idx: int = 1, same: bool = False) -> torch.Tensor:
        with torch.no_grad():
            _output = self(_input)
        target = _output.argsort(dim=-1, descending=True)[:, idx]
        if same:
            target = repeat_to_batch(target.mode(dim=0)[0], len(_input))
        return target


def add_argument(parser: argparse.ArgumentParser, model_name: str = None, model: Union[str, Model] = None,
                 config: Config = config, class_dict: dict[str, type[Model]] = {}) -> argparse._ArgumentGroup:
    dataset_name = get_name(arg_list=['-d', '--dataset'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']
    model_name = get_model_class(model_name, class_dict=class_dict)

    group = parser.add_argument_group('{yellow}model{reset}'.format(**ansi), description=model_name)
    model_class_name = get_model_class(model_name, class_dict=class_dict)
    try:
        ModelType = class_dict[model_class_name]
    except KeyError as e:
        print(f'{model_class_name} not in \n{list(class_dict.keys())}')
        raise e
    return ModelType.add_argument(group)


def create(model_name: str = None, model: Union[str, Model] = None,
           dataset_name: str = None, dataset: Union[str, Dataset] = None,
           folder_path: str = None,
           config: Config = config, class_dict: dict[str, type[Model]] = {}, **kwargs) -> Model:
    dataset_name = get_name(name=dataset_name, module=dataset, arg_list=['-d', '--dataset'])
    model_name = get_name(name=model_name, module=model, arg_list=['-m', '--model'])
    if dataset_name is None:
        dataset_name = config.get_full_config()['dataset']['default_dataset']
    if model_name is None:
        model_name = config.get_config(dataset_name=dataset_name)['model']['default_model']
    result = config.get_config(dataset_name=dataset_name)['model']._update(kwargs)
    model_name = model_name if model_name is not None else result['default_model']

    name_list = [name for sub_list in get_available_models(class_dict=class_dict).values()
                 for name in sub_list]
    name_list = sorted(name_list)
    assert model_name in name_list, f'{model_name} not in \n{name_list}'
    model_class_name = get_model_class(model_name, class_dict=class_dict)
    try:
        ModelType = class_dict[model_class_name]
    except KeyError as e:
        print(f'{model_class_name} not in \n{list(class_dict.keys())}')
        raise e
    if folder_path is None and isinstance(dataset, Dataset):
        folder_path = os.path.join(result['model_dir'], dataset.data_type, dataset.name)
    return ModelType(name=model_name, dataset=dataset, folder_path=folder_path, **result)


def get_available_models(class_dict: dict[str, type[Model]] = {}) -> dict[str, list[str]]:
    return {k: v.available_models for k, v in class_dict.items()}


def output_available_models(class_dict: dict[str, type[Model]] = {}, indent: int = 0) -> None:
    names_dict = get_available_models(class_dict)
    for k in sorted(names_dict.keys()):
        prints('{yellow}{k}{reset}'.format(k=k, **ansi), indent=indent)
        prints(names_dict[k], indent=indent + 10)
        print()


def get_model_class(name: str, class_dict: dict[str, type[Model]] = {}) -> str:
    correct_name: str = None
    for class_name in class_dict.keys():
        if class_name in name.lower() \
                and (correct_name is None or len(class_name) > len(correct_name)):
            correct_name = class_name
    if correct_name is not None:
        return correct_name
    raise KeyError(f'{name} not in {list(class_dict.keys())}')
