#!/usr/bin/env python3

from trojanzoo.utils.fim.kfac import KFAC
from trojanzoo.utils import empty_cache
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.model import accuracy, activate_params
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter, prints
from trojanzoo.environ import env

import torch
import torch.nn as nn

from tqdm import tqdm

from collections.abc import Callable
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data


def train(module: nn.Module, num_classes: int,
          epoch: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
          grad_clip: float = None, kfac: KFAC = None,
          print_prefix: str = 'Epoch', start_epoch: int = 0, resume: int = 0,
          validate_interval: int = 10, save: bool = False, amp: bool = False,
          loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
          epoch_fn: Callable[..., None] = None,
          get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
          loss_fn: Callable[..., torch.Tensor] = None,
          after_loss_fn: Callable[..., None] = None,
          validate_fn: Callable[..., tuple[float, float]] = None,
          save_fn: Callable[..., None] = None, file_path: str = None, folder_path: str = None, suffix: str = None,
          writer=None, main_tag: str = 'train', tag: str = '',
          accuracy_fn: Callable[..., list[float]] = None,
          verbose: bool = True, indent: int = 0,
          change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch', **kwargs) -> None:
    if epoch <= 0:
        return
    get_data_fn = get_data_fn if get_data_fn is not None else lambda x: x
    loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    validate_fn = validate_fn if callable(validate_fn) else validate
    accuracy_fn = accuracy_fn if callable(accuracy_fn) else accuracy

    scaler: torch.cuda.amp.GradScaler = None
    if not env['num_gpus']:
        amp = False
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    if validate_interval != 0:
        _, best_acc = validate_fn(loader=loader_valid, get_data_fn=get_data_fn, loss_fn=loss_fn,
                                  writer=None, tag=tag, _epoch=start_epoch,
                                  verbose=verbose, indent=indent, **kwargs)

    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])
    len_loader_train = len(loader_train)
    total_iter = (epoch - resume) * len_loader_train

    if resume and lr_scheduler:
        for _ in range(resume):
            lr_scheduler.step()
    for _epoch in range(resume, epoch):
        _epoch += 1
        if callable(epoch_fn):
            activate_params(module, [])
            epoch_fn(optimizer=optimizer, lr_scheduler=lr_scheduler,
                     _epoch=_epoch, epoch=epoch, start_epoch=start_epoch)
            activate_params(module, params)
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
        if change_train_eval:
            module.train()
        activate_params(module, params)
        optimizer.zero_grad()
        for i, data in enumerate(loader_epoch):
            _iter = _epoch * len_loader_train + i
            # data_time.update(time.perf_counter() - end)
            _input, _label = get_data_fn(data, mode='train')
            if kfac is not None and not amp:
                kfac.track.enable()
            _output = module(_input, amp=amp)
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
                if callable(after_loss_fn):
                    after_loss_fn(_input=_input, _label=_label, _output=_output,
                                  loss=loss, optimizer=optimizer, loss_fn=loss_fn,
                                  amp=amp, scaler=scaler,
                                  _iter=_iter, total_iter=total_iter)
                    # start_epoch=start_epoch, _epoch=_epoch, epoch=epoch)
                if kfac is not None:
                    kfac.track.disable()
                    kfac.step()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
            if lr_scheduler and lr_scheduler_freq == 'step':
                lr_scheduler.step()
            optimizer.zero_grad()
            acc1, acc5 = accuracy_fn(_output, _label, num_classes=num_classes, topk=(1, 5))
            batch_size = int(_label.size(0))
            logger.meters['loss'].update(float(loss), batch_size)
            logger.meters['top1'].update(acc1, batch_size)
            logger.meters['top5'].update(acc5, batch_size)
            empty_cache()   # TODO: should it be outside of the dataloader loop?
        if lr_scheduler and lr_scheduler_freq == 'epoch':
            lr_scheduler.step()
        if change_train_eval:
            module.eval()
        activate_params(module, [])
        loss, acc = logger.meters['loss'].global_avg, logger.meters['top1'].global_avg
        if writer is not None:
            from torch.utils.tensorboard import SummaryWriter
            assert isinstance(writer, SummaryWriter)
            writer.add_scalars(main_tag='Loss/' + main_tag, tag_scalar_dict={tag: loss},
                               global_step=_epoch + start_epoch)
            writer.add_scalars(main_tag='Acc/' + main_tag, tag_scalar_dict={tag: acc},
                               global_step=_epoch + start_epoch)
        if validate_interval != 0:
            if _epoch % validate_interval == 0 or _epoch == epoch:
                _, cur_acc = validate_fn(module=module, num_classes=num_classes,
                                         loader=loader_valid, get_data_fn=get_data_fn, loss_fn=loss_fn,
                                         writer=writer, tag=tag, _epoch=_epoch + start_epoch,
                                         verbose=verbose, indent=indent, **kwargs)
                if cur_acc >= best_acc:
                    if verbose:
                        prints('{purple}best result update!{reset}'.format(**ansi), indent=indent)
                        prints(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}', indent=indent)
                    best_acc = cur_acc
                    if save:
                        save_fn(file_path=file_path, folder_path=folder_path, suffix=suffix, verbose=verbose)
                if verbose:
                    prints('-' * 50, indent=indent)
    module.zero_grad()


def validate(module: nn.Module, num_classes: int, loader: torch.utils.data.DataLoader,
             print_prefix: str = 'Validate', indent: int = 0, verbose: bool = True,
             get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
             loss_fn: Callable[..., torch.Tensor] = None,
             writer=None, main_tag: str = 'valid', tag: str = '', _epoch: int = None,
             accuracy_fn: Callable[..., list[float]] = None,
             **kwargs) -> tuple[float, float]:
    module.eval()
    get_data_fn = get_data_fn if get_data_fn is not None else lambda x: x
    loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    accuracy_fn = accuracy_fn if callable(accuracy_fn) else accuracy
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
            _output = module(_input)
            loss = float(loss_fn(_input, _label, _output=_output, **kwargs))
            acc1, acc5 = accuracy_fn(_output, _label, num_classes=num_classes, topk=(1, 5))
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


def compare(module1: nn.Module, module2: nn.Module, loader: torch.utils.data.DataLoader,
            print_prefix='Validate', indent=0, verbose=True,
            get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs) -> float:
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    module1.eval()
    module2.eval()
    get_data_fn = get_data_fn if get_data_fn is not None else lambda x: x

    def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = -softmax(p) * logsoftmax(q)
        return result.sum(1).mean()

    logger = MetricLogger()
    logger.meters['loss'] = SmoothedValue()
    loader_epoch = loader
    if verbose:
        header = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
        if env['tqdm']:
            header = '{upline}{clear_line}'.format(**ansi) + header
            loader_epoch = tqdm(loader_epoch)
        loader_epoch = logger.log_every(loader_epoch, header=header, indent=indent)
    with torch.no_grad():
        for data in loader_epoch:
            _input, _label = get_data_fn(data, **kwargs)
            _output1, _output2 = module1(_input), module2(_input)
            loss = float(cross_entropy(_output1, _output2))
            batch_size = int(_label.size(0))
            logger.meters['loss'].update(loss, batch_size)
    return logger.meters['loss'].global_avg
