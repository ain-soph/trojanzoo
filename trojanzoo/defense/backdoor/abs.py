# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor
from trojanzoo.utils import to_tensor, repeat_to_batch
from trojanzoo.utils.model import total_variation
from trojanzoo.utils.output import prints, ansi, output_iter
from trojanzoo.utils.ssim import SSIM
from trojanzoo.optim.uname import Uname

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import math

from typing import Dict

from trojanzoo.utils import Config
env = Config.env


class ABS(Defense_Backdoor):

    name: str = 'abs'

    def __init__(self, seed_num: int = 50,
                 samp_k: int = 1, same_range: bool = False, n_samples: int = 5,
                 max_troj_size: int = 16, re_mask_lr: float = 0.1, re_mask_weight: float = 500, re_iteration: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.seed_num: int = seed_num

        # -----------Neural Sampling------------- #
        self.samp_k: int = samp_k
        self.same_range: bool = same_range
        self.n_samples: int = n_samples
        self.top_n_neurons: int = 20

        # ----------------Remask----------------- #
        self.max_troj_size: int = max_troj_size
        self.re_mask_lr: float = re_mask_lr
        self.re_mask_weight: float = re_mask_weight
        self.re_iteration: int = re_iteration

        self.ssim = SSIM()

    def detect(self, **kwargs):
        super().detect(**kwargs)
        mark_list, mask_list, loss_list = self.get_potential_triggers()

    def get_potential_triggers(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        seed_data = self.load_seed_data()
        _input, _label = seed_data['input'], seed_data['label']
        all_ps = self.sample_neuron(_input)
        neuron_dict = self.find_min_max(all_ps, _label)

        mark_list, mask_list, entropy_list = [], [], []
        for layer, layer_dict in neuron_dict.items():
            for neuron, label in neuron_dict.items():
                self.remask(layer=layer, neuron=neuron)

        return mark_list, mask_list, entropy_list

    def remask(self, layer: str, neuron: int):
        # no bound
        atanh_mark = torch.randn(self.data_shape, device=env['device'])
        atanh_mark.requires_grad = True
        atanh_mask = torch.randn(self.data_shape[1:], device=env['device'])
        atanh_mask.requires_grad = True
        mask = Uname.tanh_func(atanh_mask)    # (h, w)
        mark = Uname.tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam([atanh_mark, atanh_mask], lr=0.2)
        optimizer.zero_grad()

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        loss_best = None

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        for _epoch in range(nc_epoch):
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            for data in tqdm(self.dataset.loader['train']):
                _input, _label = self.model.get_data(data)
                batch_size = _label.size(0)
                X = _input + mask * (mark - _input)
                Y = label * torch.ones_like(_label, dtype=torch.long)
                _output = self.model(X)

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = self.model.criterion(_output, Y)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = Uname.tanh_func(atanh_mask)    # (h, w)
                mark = Uname.tanh_func(atanh_mark)    # (c, h, w)
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = '{blue_light}Epoch: {0}'.format(
                output_iter(_epoch + 1, nc_epoch), **ansi).ljust(60)
            _str = ' '.join([
                'Loss: {:.4f},'.format(losses.avg).ljust(20),
                'Acc: {:.2f}, '.format(acc.avg).ljust(20),
                'Norm: {:.4f},'.format(norm.avg).ljust(20),
                'Entropy: {:.4f},'.format(entropy.avg).ljust(20),
                'Time: {},'.format(epoch_time).ljust(20),
            ])
            prints(pre_str, _str, prefix='\033[1A\033[K', indent=4)

        self.attack.mark.mark = mark
        self.attack.mark.alpha_mark = mask
        self.attack.mark.mask = torch.ones_like(mark, dtype=torch.bool)
        self.attack.validate_func()
        return mark_best, mask_best, entropy_best

    # ---------------------------- Seed Data --------------------------- #
    def save_seed_data(self) -> Dict[str, np.ndarray]:
        torch.manual_seed(env['seed'])
        if self.seed_num % self.model.num_classes:
            raise ValueError('seed_num({0:d}) % num_classes({1:d}) should be 0.'.format(
                self.seed_num, self.model.num_classes))
        seed_class_num: int = self.seed_num // self.model.num_classes
        x, y = [], []
        for _class in range(self.model.num_classes):
            loader = self.dataset.get_dataloader(mode='train', batch_size=seed_class_num,
                                                 shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            _input, _label = next(iter(loader))
            x.append(_input)
            y.append(_label)
        x = torch.stack(x).flatten(end_dim=1).numpy()
        y = torch.stack(y).flatten(end_dim=1).numpy()
        seed_data = {'input': x, 'label': y}
        seed_path = env['result_dir'] + '{0:s}/{1:s}_{2:d}.npy'.format(self.name, self.dataset.name, self.seed_num)
        np.save(seed_path, seed_data)
        print('seed data saved at: ', seed_path)
        return seed_data

    def load_seed_data(self) -> Dict[str, torch.Tensor]:
        seed_path = env['result_dir'] + '{0:s}/{1:s}_{2:d}.npy'.format(self.name, self.dataset.name, self.seed_num)
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

            vs = self.samp_k * torch.arange(self.n_samples, device=h_t.device)
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
            result = self.model.get_layer(h_t.flatten(end_dim=2), layer_input=layer).detach().cpu()
            result_shape = list(h_t.shape)[:3]
            result_shape.extend(list(result.shape)[1:])
            result = result.view(result_shape)
            # (C, n_samples, batch_size, num_classes)
            all_ps[layer] = result
        return all_ps

    def find_min_max(self, all_ps: Dict[str, torch.Tensor], _label: torch.Tensor) -> Dict[str, Dict[int, float]]:
        neuron_dict: Dict[str, Dict[int, float]] = {}
        for layer in all_ps.keys():
            ps = all_ps[layer]  # (C, n_samples, batch_size, num_classes)
            vs: torch.Tensor = ps[:, self.n_samples // 5:].max(dim=1)[0] \
                - ps[:, :self.n_samples // 5].min(dim=1)[0]  # (C, batch_size, num_classes)
            values, labels = vs.sort(dim=-1, descending=True)
            values = values[:, :, 0] - values[:, :, 1]  # (C, batch_size)
            labels = labels[:, :, 0]  # (C, batch_size)

            mode_labels = labels[:, :, 0].mode(keepdim=True)[0]  # (C, 1)
            _labels = _labels.view(1, -1)  # (1, batch_size)
            other_idx1 = ~_labels.eq(mode_labels)  # (C, batch_size)
            other_idx = torch.bitwise_and(other_idx1, labels.eq(_labels))  # (C, batch_size)
            condition1 = other_idx.sum(dim=-1, keepdim=True)  # (C, 1)
            other_idx = torch.where(condition1, other_idx, other_idx1)  # (C, batch_size)

            min_values, min_idx = torch.where(other_idx, values, values.max()).min(dim=-1)[0]  # (C)
            min_labels = labels.gather(dim=1, index=min_idx.unsqueeze(1)).flatten()  # (C)
            min_labels_counts = labels.eq(min_labels.unsqueeze(1)).int().sum(dim=1)  # (C)
            condition2 = min_labels.ge(self.n_samples - 2)   # todo: Not sure: self.n_samples -> self.seed_num
            idx_list = condition2.nonzero()[:self.top_n_neurons]
            neuron_dict[layer] = {int(idx): int(min_labels[idx]) for idx in idx_list}
        return neuron_dict
    # -------------------------ReMask--------------------------------- #

    def abs_loss(self, _input: torch.Tensor, atanh_mark: torch.Tensor, atanh_mask: torch.Tensor,
                 layer: str, neuron: int, next_neuron: int):
        mark = atanh_mark.tanh().mul(0.5).add(0.5)
        mask = atanh_mask.tanh().mul(0.5).add(0.5) * self.nc_mask

        X = _input + mask * (mark - _input)
        _dict: Dict[str, torch.Tensor] = self.model.get_all_layer(X)
        tinners = _dict[layer]
        logits = _dict['logits']

        vloss1 = tinners[:, neuron].sum()
        vloss2 = tinners.sum() - vloss1
        tvloss = total_variation(mark)

        mask_loss = mask.sum()

        loss = -vloss1 + 1e-4 * vloss2
        if mask_loss > self.max_troj_size:
            pass

        ssim_loss = - self.ssim()  # todo
        ssim_loss *= 10 if ssim_loss < -2 else 10000
        loss = -vloss1 + 1e-5 * vloss2 + 1e-3 * tvloss
        loss = 0.01 * loss + ssim_loss
        return loss

    # ---------------------------------- Utils ------------------------------- #
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
