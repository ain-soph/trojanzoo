#!/usr/bin/env python3

from ..backdoor_defense import BackdoorDefense
from trojanvision.environ import env
from trojanvision.utils.loss import total_variation
from trojanvision.utils.ssim import SSIM
from trojanzoo.utils import normalize_mad, jaccard_idx
from trojanzoo.utils import to_tensor, to_numpy, tanh_func
from trojanzoo.utils import AverageMeter
from trojanzoo.utils.output import prints, ansi, output_iter

import torch
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import datetime


class ABS(BackdoorDefense):
    name: str = 'abs'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--seed_num', type=int, help='ABS seed number, defaults to -5.')
        group.add_argument('--max_troj_size', type=int,
                           help='ABS max trojan trigger size (pixel number), defaults to 64.')
        group.add_argument('--remask_epoch', type=int, help='ABS optimizing epoch, defaults to 1000.')
        group.add_argument('--remask_lr', type=float, help='ABS optimization learning rate, defaults to 0.1.')
        group.add_argument('--remask_weight', type=float, help='ABS optimization remask loss weight, defaults to 0.1.')
        return group

    # todo: use_mask=False case, X=(1-mask)*_input + mask* mark is not correct.
    # It is the weighted average among _input, maxpool, minpool and avgpool.
    # See filter_load_model
    def __init__(self, seed_num: int = -5, count_mask: bool = True,
                 samp_k: int = 1, same_range: bool = False, n_samples: int = 5,
                 max_troj_size: int = 16, remask_lr: float = 0.1, remask_epoch: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.param_list['abs'] = ['seed_num', 'count_mask', 'samp_k', 'same_range', 'n_samples',
                                  'max_troj_size', 'remask_lr', 'remask_epoch']

        self.seed_num: int = seed_num
        if self.seed_num < 0:
            self.seed_num = self.model.num_classes * abs(self.seed_num)
        self.count_mask: bool = count_mask

        # -----------Neural Sampling------------- #
        self.samp_k: int = samp_k
        self.same_range: bool = same_range
        self.n_samples: int = n_samples
        self.top_n_neurons: int = 20

        # ----------------Remask----------------- #
        self.max_troj_size: int = max_troj_size
        self.remask_lr: float = remask_lr
        self.remask_epoch: int = remask_epoch

        self.ssim = SSIM()
        # self.nc_mask = self.nc_filter_img()

    def detect(self, **kwargs):
        super().detect(**kwargs)
        if not self.attack.mark.random_pos:
            self.real_mask = self.attack.mark.mask
        seed_data = self.load_seed_data()
        _input, _label = seed_data['input'], seed_data['label']
        print('sample neurons')
        all_ps = self.sample_neuron(_input)
        print('find min max')
        neuron_dict = self.find_min_max(all_ps, _label)
        self.print_neuron_dict(neuron_dict)
        print()
        print()
        print('remask')
        self.get_potential_triggers(neuron_dict, _input, _label)

    def print_neuron_dict(self, neuron_dict: dict[int, list[dict]]):
        for label, label_list in neuron_dict.items():
            print('label: ', label)
            for _dict in label_list:
                layer = _dict['layer']
                neuron = _dict['neuron']
                value = _dict['value']
                _str = f'    layer: {layer:20s}    neuron: {neuron:5d}    value: {value:.3f}'
                if 'loss' in _dict.keys():
                    loss = _dict['loss']
                    attack_acc = _dict['attack_acc']
                    _str += f'    loss: {loss:10.3f}    Attack Acc: {attack_acc:.3f}'
                print(_str)

    def get_potential_triggers(self, neuron_dict: dict[int, list[dict]], _input: torch.Tensor, _label: torch.Tensor, use_mask=True) -> dict[int, list[dict]]:
        losses = AverageMeter('Loss', ':.4e')
        norms = AverageMeter('Norm', ':6.2f')
        jaccard = AverageMeter('Jaccard Idx', ':6.2f')
        score_list = [0.0] * len(list(neuron_dict.keys()))
        result_dict = {}
        self.attack.mark.random_pos = False
        self.attack.mark.height_offset = 0
        self.attack.mark.width_offset = 0
        for label, label_list in neuron_dict.items():
            print('label: ', label)
            best_score = 1e7
            for _dict in reversed(label_list):
                layer = _dict['layer']
                neuron = _dict['neuron']
                value = _dict['value']
                # color = ('{red}' if label == self.attack.target_class else '{green}').format(**ansi)
                # _str = f'layer: {layer:<20} neuron: {neuron:<5d} label: {label:<5d}'
                # prints('{color}{_str}{reset}'.format(color=color, _str=_str, **ansi), indent=4)
                mark, mask, loss = self.remask(_input, layer=layer, neuron=neuron,
                                               label=label, use_mask=use_mask)
                self.attack.mark.mark = mark
                self.attack.mark.alpha_mask = mask
                self.attack.mark.mask = torch.ones_like(mark, dtype=torch.bool)
                attack_loss, attack_acc = self.model._validate(
                    verbose=False, get_data_fn=self.attack.get_data,
                    keep_org=False, poison_label=True)
                _dict['loss'] = loss
                _dict['attack_acc'] = attack_acc
                _dict['attack_loss'] = attack_loss
                _dict['mask'] = to_numpy(mask)
                _dict['mark'] = to_numpy(mark)
                _dict['norm'] = float(mask.norm(p=1))
                score = attack_loss + 7e-2 * float(mask.norm(p=1))
                if score < best_score:
                    best_score = score
                    result_dict[label] = _dict
                if attack_acc > 90:
                    losses.update(loss)
                    norms.update(mask.norm(p=1))
                _str = f'    layer: {layer:20s}    neuron: {neuron:5d}    value: {value:.3f}'
                _str += f'    loss: {loss:10.3f}'
                _str += f'    ATK Acc: {attack_acc:.3f}'
                _str += f'    ATK Loss: {attack_loss:10.3f}'
                _str += f'    Norm: {mask.norm(p=1):.3f}'
                _str += f'    Score: {score:.3f}'
                if not self.attack.mark.random_pos:
                    overlap = jaccard_idx(mask, self.real_mask)
                    _dict['jaccard'] = overlap
                    _str += f'    Jaccard: {overlap:.3f}'
                    if attack_acc > 90:
                        jaccard.update(overlap)
                else:
                    _dict['jaccard'] = 0.0
                print(_str)
                if not os.path.exists(self.folder_path):
                    os.makedirs(self.folder_path)
                np.save(os.path.join(self.folder_path,
                                     self.get_filename(target_class=self.target_class) + '.npy'), neuron_dict)
                np.save(os.path.join(self.folder_path,
                                     self.get_filename(target_class=self.target_class) + '_best.npy'), result_dict)
            print(f'    Label: {label:3d}  loss: {result_dict[label]["loss"]:10.3f}'
                  f'  ATK Acc: {result_dict[label]["attack_acc"]:.3f}'
                  f'  ATK loss: {result_dict[label]["attack_loss"]:10.3f}  Norm: {result_dict[label]["norm"]:10.3f}'
                  f'  Jaccard: {result_dict[label]["jaccard"]:10.3f}  Score: {best_score:.3f}')
            score_list[label] = best_score
        print('Score: ', score_list)
        print('Score MAD: ', normalize_mad(score_list))
        return neuron_dict

    def remask(self, _input: torch.Tensor, layer: str, neuron: int,
               label: int, use_mask: bool = True, validate_interval: int = 100,
               verbose=False) -> tuple[torch.Tensor, torch.Tensor, float]:
        atanh_mark = torch.randn(self.dataset.data_shape, device=env['device'])
        atanh_mark.requires_grad_()
        parameters: list[torch.Tensor] = [atanh_mark]
        mask = torch.ones(self.dataset.data_shape[1:], device=env['device'])
        atanh_mask = torch.ones(self.dataset.data_shape[1:], device=env['device'])
        if use_mask:
            atanh_mask.requires_grad_()
            parameters.append(atanh_mask)
            mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam(parameters, lr=self.remask_lr if use_mask else 0.01 * self.remask_lr)
        optimizer.zero_grad()

        # best optimization results
        mark_best = None
        loss_best = float('inf')
        mask_best = None

        for _epoch in range(self.remask_epoch):
            epoch_start = time.perf_counter()

            loss = self.abs_loss(_input, mask, mark, layer=layer, neuron=neuron, use_mask=use_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mark = tanh_func(atanh_mark)    # (c, h, w)
            if use_mask:
                mask = tanh_func(atanh_mask)    # (h, w)

            with torch.no_grad():
                X = _input + mask * (mark - _input)
                _output = self.model(X)
            acc = float(_output.argmax(dim=1).eq(label).float().mean()) * 100
            loss = float(loss)

            if verbose:
                norm = mask.norm(p=1)
                epoch_time = str(datetime.timedelta(seconds=int(
                    time.perf_counter() - epoch_start)))
                pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                    output_iter(_epoch + 1, self.remask_epoch), **ansi).ljust(64 if env['color'] else 35)
                _str = ' '.join([
                    f'Loss: {loss:10.3f},'.ljust(20),
                    f'Acc: {acc:.3f}, '.ljust(20),
                    f'Norm: {norm:.3f},'.ljust(20),
                    f'Time: {epoch_time},'.ljust(20),
                ])
                prints(pre_str, _str, indent=8)
            if loss < loss_best:
                loss_best = loss
                mark_best = mark
                if use_mask:
                    mask_best = mask
            if validate_interval != 0 and verbose:
                if (_epoch + 1) % validate_interval == 0 or _epoch == self.remask_epoch - 1:
                    self.attack.mark.mark = mark
                    self.attack.mark.alpha_mask = mask
                    self.attack.mark.mask = torch.ones_like(mark, dtype=torch.bool)
                    self.attack.target_class = label
                    self.model._validate(print_prefix='Validate Trigger Tgt',
                                         get_data_fn=self.attack.get_data,
                                         keep_org=False, poison_label=True, indent=8)
                    print()
        atanh_mark.requires_grad = False
        if use_mask:
            atanh_mask.requires_grad = False
        return mark_best, mask_best, loss_best

    # ---------------------------- Seed Data --------------------------- #
    def save_seed_data(self) -> dict[str, np.ndarray]:
        torch.manual_seed(env['seed'])
        if self.seed_num % self.model.num_classes:
            raise ValueError(f'seed_num({self.seed_num:d}) % num_classes({self.model.num_classes:d}) should be 0.')
        seed_class_num: int = self.seed_num // self.model.num_classes
        x, y = [], []
        for _class in range(self.model.num_classes):
            loader = self.dataset.get_dataloader(mode='train', batch_size=seed_class_num, class_list=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            x.append(_input)
            y.append(_label)
        x = torch.cat(x).numpy()
        y = torch.cat(y).numpy()
        seed_data = {'input': x, 'label': y}
        seed_path = os.path.join(self.folder_path, f'seed_{self.seed_num}.npy')
        np.save(seed_path, seed_data)
        print('seed data saved at: ', seed_path)
        return seed_data

    def load_seed_data(self) -> dict[str, torch.Tensor]:
        seed_path = os.path.join(self.folder_path, f'seed_{self.seed_num}.npy')
        seed_data: dict[str, torch.Tensor] = {}
        seed_data = np.load(seed_path, allow_pickle=True).item() if os.path.exists(seed_path) \
            else self.save_seed_data()
        seed_data['input'] = to_tensor(seed_data['input'])
        seed_data['label'] = to_tensor(seed_data['label'], dtype=torch.long)
        return seed_data

    # -----------------------Neural Sample---------------------------- #

    def sample_neuron(self, _input: torch.Tensor) -> dict[str, torch.Tensor]:
        all_ps: dict[str, torch.Tensor] = {}
        layer_output = self.model.get_all_layer(_input)
        for layer in layer_output.keys():
            if not layer.startswith('features.') and not layer.startswith('classifier.'):
                continue
            cur_layer_output: torch.Tensor = layer_output[layer].detach().cpu()  # (batch_size, C, H, W)
            channel_num: int = cur_layer_output.shape[1]  # channels

            h_t: torch.Tensor = cur_layer_output.expand([channel_num, self.n_samples] + [-1] * cur_layer_output.dim())
            # (C, n_samples, batch_size, C, H, W)

            vs = self.samp_k * torch.arange(self.n_samples, device=h_t.device, dtype=torch.float)
            if not self.same_range:
                maxes = cur_layer_output.max()
                vs *= float(maxes) / self.n_samples
            vs_shape = [1] * cur_layer_output.dim()
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
                h = h.to(device=env['device'])
                result.append(self.model.get_layer(h.flatten(end_dim=1), layer_input=layer).detach().cpu())
            result = torch.cat(result)

            result_shape = list(h_t.shape)[:3]
            result_shape.extend(list(result.shape)[1:])
            result = result.view(result_shape)
            all_ps[layer] = result
            # (C, n_samples, batch_size, num_classes)
        return all_ps

    def find_min_max(self, all_ps: dict[str, torch.Tensor], _label: torch.Tensor) -> dict[int, list[dict]]:
        neuron_dict: dict[int, list] = {i: [] for i in range(self.model.num_classes)}
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
        # X = self.attack.add_mark(_input)
        X = _input + mask * (mark - _input)
        # return self.model.loss(X, torch.zeros(X.size(0), device=X.device, dtype=torch.long)) + 1e-3 * mask.norm(p=1)
        feats = self.model.get_layer(X, layer_output=layer)
        # if feats.dim() > 2:
        #     feats = feats.flatten(2).sum(2)
        # from torch.nn.functional import log_softmax
        # feats = log_softmax(feats)
        vloss1 = feats[:, neuron].sum()
        vloss2 = feats.sum() - vloss1
        loss = torch.zeros_like(vloss1)
        if use_mask:
            mask_loss = mask.sum()
            if mask_loss > 100:
                mask_loss *= 100
            if mask_loss > self.max_troj_size:
                mask_loss *= 10
            else:
                mask_loss *= 1e-1
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
        h, w = self.dataset.data_shape[1:]
        mask = torch.zeros(h, w, dtype=torch.float)
        mask[2:7, 2:7] = 1
        return to_tensor(mask, non_blocking=False)

    def nc_filter_img(self) -> torch.Tensor:
        h, w = self.dataset.data_shape[1:]
        mask = torch.ones(h, w, dtype=torch.float)
        return to_tensor(mask, non_blocking=False)
        # TODO: fix
        # mask = torch.zeros(h, w, dtype=torch.float)
        # if self.use_mask:
        #     mask[math.ceil(0.25 * w): math.floor(0.75 * w), math.ceil(0.25 * h): math.floor(0.75 * h)] = 1
        # else:
        #     mask.add_(1)

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.folder_path, self.get_filename(target_class=self.target_class) + '_best.npy')
        _dict: dict[str, dict[str, torch.Tensor]] = np.load(path, allow_pickle=True).item()
        self.attack.mark.mark = to_tensor(_dict[self.target_class]['mark'])
        self.attack.mark.alpha_mask = to_tensor(_dict[self.target_class]['mask'])
        self.attack.mark.mask = torch.ones_like(self.attack.mark.mark, dtype=torch.bool)
        self.attack.mark.random_pos = False
        self.attack.mark.height_offset = 0
        self.attack.mark.width_offset = 0
        print('defense results loaded from: ', path)
