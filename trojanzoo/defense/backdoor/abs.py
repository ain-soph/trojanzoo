# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor
from trojanzoo.utils import to_tensor
from trojanzoo.utils.model import AverageMeter, total_variation
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.utils.ssim import SSIM
from trojanzoo.optim.uname import Uname

import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import datetime
import numpy as np
import os
import math

from typing import Dict

from trojanzoo.utils.config import Config
env = Config.env


class ABS(Defense_Backdoor):
    name: str = 'abs'

    # todo: use_mask=False case, X=(1-mask)*_input + mask* mark is not correct.
    # It is the weighted average among _input, maxpool, minpool and avgpool.
    # See filter_load_model
    def __init__(self, seed_num: int = 50, count_mask: bool = True,
                 samp_k: int = 1, same_range: bool = False, n_samples: int = 5,
                 max_troj_size: int = 64, remask_lr: float = 0.1, remask_weight: float = 500, remask_epoch: int = 1000, **kwargs):
        super().__init__(**kwargs)
        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape

        self.seed_num: int = seed_num
        self.count_mask: bool = count_mask

        # -----------Neural Sampling------------- #
        self.samp_k: int = samp_k
        self.same_range: bool = same_range
        self.n_samples: int = n_samples
        self.top_n_neurons: int = 20

        # ----------------Remask----------------- #
        self.max_troj_size: int = max_troj_size
        self.remask_lr: float = remask_lr
        self.remask_weight: float = remask_weight
        self.remask_epoch: int = remask_epoch

        self.ssim = SSIM()
        self.nc_mask = self.nc_filter_img()

    def detect(self, **kwargs):
        super().detect(**kwargs)
        seed_data = self.load_seed_data()
        _input, _label = seed_data['input'], seed_data['label']
        print('sample neurons')
        all_ps = self.sample_neuron(_input)
        print('find min max')
        neuron_dict = self.find_min_max(all_ps, _label)
        print('remask')
        neuron_dict = self.get_potential_triggers(neuron_dict, _input, _label)
        loss_list = []
        for label, label_list in neuron_dict.items():
            for element in label_list:
                loss_list.append(element['loss'])
        

    def get_potential_triggers(self, neuron_dict: Dict[str, Dict[int, int]], _input: torch.Tensor, _label: torch.LongTensor, use_mask=True) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        for label, label_list in neuron_dict.items():
            for _dict in label_list:
                layer = _dict['layer']
                neuron = _dict['neuron']
                color = ('{red}' if label == self.attack.target_class else '{green}').format(**ansi)
                _str = f'layer: {layer:<20} neuron: {neuron:<5d} label: {label:<5d}'
                prints('{color}{_str}{reset}'.format(color=color, _str=_str, **ansi), indent=4)
                mark, mask, loss = self.remask(_input, layer=layer, neuron=neuron,
                                               label=label, use_mask=use_mask)
                _dict['loss'] = loss

        return neuron_dict

    def remask(self, _input: torch.Tensor, layer: str, neuron: int,
               label: int, use_mask: bool = True, validate_interval: int = 100) -> (torch.Tensor, torch.Tensor, float):
        atanh_mark = torch.randn(self.data_shape, device=env['device'])
        atanh_mark.requires_grad_()
        parameters: List[torch.Tensor] = [atanh_mark]
        mask = torch.ones(self.data_shape[1:], device=env['device'])
        if use_mask:
            atanh_mask = torch.randn(self.data_shape[1:], device=env['device'])
            atanh_mask.requires_grad_()
            parameters.append(atanh_mask)
            mask = Uname.tanh_func(atanh_mask) * self.nc_mask    # (h, w)
        mark = Uname.tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam(parameters, lr=self.remask_lr if use_mask else 0.01 * self.remask_lr)
        optimizer.zero_grad()

        # best optimization results
        norm_best = float('inf')
        mark_best = None
        loss_best = float('inf')
        mask_best = None

        batch_size = _input.size(0)
        print()
        for _epoch in range(self.remask_epoch):
            epoch_start = time.perf_counter()

            loss = self.abs_loss(_input, mask, mark, layer=layer, neuron=neuron, use_mask=use_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mark = Uname.tanh_func(atanh_mark)    # (c, h, w)
            if use_mask:
                mask = Uname.tanh_func(atanh_mask) * self.nc_mask    # (h, w)

            with torch.no_grad():
                X = _input + mask * (mark - _input)
                _output = self.model(X)
            acc = float(_output.argmax(dim=1).eq(label).float().mean()) * 100
            norm = mask.norm(p=1)
            loss = float(loss)

            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                output_iter(_epoch + 1, self.remask_epoch), **ansi).ljust(64 if env['color'] else 35)
            _str = ' '.join([
                f'Loss: {loss:.4f},'.ljust(20),
                f'Acc: {acc:.3f}, '.ljust(20),
                f'Norm: {norm:.4f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            prints(pre_str, _str, indent=8)
            if loss < loss_best:
                loss_best = loss
                mark_best = mark
                if use_mask:
                    mask_best = mask
            if validate_interval != 0:
                if (_epoch + 1) % validate_interval == 0 or _epoch == self.remask_epoch - 1:
                    self.attack.mark.mark = mark
                    self.attack.mark.alpha_mark = mask
                    self.attack.mark.mask = torch.ones_like(mark, dtype=torch.bool)
                    self.attack.target_class = label
                    self.model._validate(print_prefix='Validate Trigger Tgt',
                                         get_data=self.attack.get_data, keep_org=False, indent=8)
                    print()
        atanh_mark.requires_grad = False
        if use_mask:
            atanh_mask.requires_grad = False
        return mark_best, mask_best, loss_best

    # ---------------------------- Seed Data --------------------------- #
    def save_seed_data(self) -> Dict[str, np.ndarray]:
        torch.manual_seed(env['seed'])
        if self.seed_num % self.model.num_classes:
            raise ValueError(f'seed_num({self.seed_num:d}) % num_classes({self.model.num_classes:d}) should be 0.')
        seed_class_num: int = self.seed_num // self.model.num_classes
        x, y = [], []
        for _class in range(self.model.num_classes):
            loader = self.dataset.get_dataloader(mode='train', batch_size=seed_class_num, classes=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            x.append(_input)
            y.append(_label)
        x = torch.cat(x).numpy()
        y = torch.cat(y).numpy()
        seed_data = {'input': x, 'label': y}
        seed_path = f'{env["result_dir"]}{self.dataset.name}/{self.name}_{self.seed_num}.npy'
        np.save(seed_path, seed_data)
        print('seed data saved at: ', seed_path)
        return seed_data

    def load_seed_data(self) -> Dict[str, torch.Tensor]:
        seed_path = f'{env["result_dir"]}{self.dataset.name}/{self.name}_{self.seed_num}.npy'
        seed_data: Dict[str, torch.Tensor] = {}
        seed_data = np.load(seed_path, allow_pickle=True).item() if os.path.exists(seed_path) \
            else self.save_seed_data()
        seed_data['input'] = to_tensor(seed_data['input'])
        seed_data['label'] = to_tensor(seed_data['label'], dtype=torch.long)
        return seed_data

    # -----------------------Neural Sample---------------------------- #

    def sample_neuron(self, _input: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_ps: Dict[str, torch.Tensor] = {}
        batch_size = _input.shape[0]

        layer_output = self.model.get_all_layer(_input)
        for layer in self.model.get_layer_name():
            if 'pool' in layer or layer in ['features', 'flatten', 'classifier', 'logits', 'output']:
                continue
            cur_layer_output: torch.Tensor = layer_output[layer]  # (batch_size, C, H, W)
            channel_num: int = cur_layer_output.shape[1]  # channels

            repeat_shape = [channel_num, self.n_samples]
            repeat_shape.extend([1] * len(cur_layer_output.shape))
            h_t: torch.Tensor = cur_layer_output.repeat(repeat_shape)
            # (C, n_samples, batch_size, C, H, W)

            vs = self.samp_k * torch.arange(self.n_samples, device=h_t.device, dtype=torch.float)
            if not self.same_range:
                maxes = cur_layer_output.max()
                vs *= float(maxes) / self.n_samples
            vs_shape = [1] * len(cur_layer_output.shape)
            vs_shape[0] = -1
            vs = vs.view(vs_shape)
            # (n_samples, 1, 1, 1)
            # todo: might use parallel to avoid for loop (torch.Tensor.scatter?)
            for neuron in range(channel_num):
                h_t[neuron, :, :, neuron] = vs
            # todo: the shape is too large
            # result = self.model.get_layer(h_t.flatten(end_dim=2), layer_input=layer).detach().cpu()
            result = []
            for h in h_t:
                result.append(self.model.get_layer(h.flatten(end_dim=1), layer_input=layer).detach().cpu())
            result = torch.cat(result)

            result_shape = list(h_t.shape)[:3]
            result_shape.extend(list(result.shape)[1:])
            result = result.view(result_shape)
            all_ps[layer] = result
            # (C, n_samples, batch_size, num_classes)
        return all_ps

    def find_min_max(self, all_ps: Dict[str, torch.Tensor], _label: torch.Tensor) -> Dict[int, list]:
        neuron_dict: Dict[int, list] = {i: [] for i in range(self.model.num_classes)}
        _label = _label.cpu()
        for layer in all_ps.keys():
            ps = all_ps[layer]  # (C, n_samples, batch_size, num_classes)
            vs: torch.Tensor = ps[:, self.n_samples // 5:].max(dim=1)[0] \
                - ps[:, :self.n_samples // 5].min(dim=1)[0]  # (C, batch_size, num_classes)
            values, labels = vs.sort(dim=-1, descending=True)
            condition1 = labels[:, :, 0].eq(_label)  # exclude the ground-truth labels
            values = torch.where(condition1, values[:, :, 1] - values[:, :, 2],
                                 values[:, :, 0] - values[:, :, 1])  # (C, batch_size)
            labels = torch.where(condition1, labels[:, :, 1], labels[:, :, 0])  # (C, batch_size)

            mode_labels = labels.mode(keepdim=True)[0]  # (C, 1) The most frequent label
            mode_idx = labels.eq(mode_labels)  # (C, batch_size)
            mode_labels_counts = mode_idx.sum(dim=-1)  # (C)
            condition2 = mode_labels_counts.ge(self.seed_num * 0.75)
            idx_list = condition2.nonzero().flatten().tolist()
            idx_list = sorted(idx_list, key=lambda idx: float(values[idx][mode_idx[idx]].min()))[:self.top_n_neurons]
            for idx in idx_list:
                value = float(values[idx][mode_idx[idx]].min())
                neuron_dict[int(mode_labels[idx])].append({'layer': layer, 'neuron': int(idx), 'value': value})
        return neuron_dict
    # -------------------------ReMask--------------------------------- #

    def abs_loss(self, _input: torch.Tensor, mask: torch.Tensor, mark: torch.Tensor,
                 layer: str, neuron: int, use_mask: bool = True):
        X = _input + mask * (mark - _input)
        feats = self.model.get_layer(X, layer_output=layer)
        vloss1 = feats[:, neuron].sum()
        vloss2 = feats.sum() - vloss1
        loss = torch.zeros_like(vloss1)
        if use_mask:
            mask_loss = mask.sum()
            mask_nz = len(mask.nonzero())
            if (self.count_mask and mask_nz > (math.sqrt(self.max_troj_size) + 2)**2) \
                    or (not self.count_mask and mask_loss > 100):
                mask_loss *= 2 * self.remask_weight
            elif (self.count_mask and mask_nz > self.max_troj_size) \
                    or (not self.count_mask and mask_loss > self.max_troj_size):
                mask_loss *= self.remask_weight
            else:
                mask_loss *= 0.0
            loss = -vloss1 + 1e-4 * vloss2 + mask_loss
        else:
            tvloss = total_variation(mark)
            ssim_loss = - self.ssim(X, _input)
            ssim_loss *= 10 if ssim_loss < -2 else 10000
            loss = -vloss1 + 1e-5 * vloss2 + 1e-3 * tvloss
            loss = 0.01 * loss + ssim_loss
        return loss

    # ---------------------------------- Utils ------------------------------- #
    # Unused
    def filter_img(self):
        h, w = self.dataset.n_dim
        mask = torch.zeros(h, w, dtype=torch.float)
        mask[2:7, 2:7] = 1
        return to_tensor(mask, non_blocking=False)

    def nc_filter_img(self) -> torch.Tensor:
        h, w = self.dataset.n_dim
        mask = torch.ones(h, w, dtype=torch.float)
        return to_tensor(mask, non_blocking=False)
        # todo: fix
        # mask = torch.zeros(h, w, dtype=torch.float)
        # if self.use_mask:
        #     mask[math.ceil(0.25 * w): math.floor(0.75 * w), math.ceil(0.25 * h): math.floor(0.75 * h)] = 1
        # else:
        #     mask.add_(1)
