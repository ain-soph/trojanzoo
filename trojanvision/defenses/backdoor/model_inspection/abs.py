#!/usr/bin/env python3

from ...abstract import ModelInspection
from trojanvision.environ import env
from trojanzoo.utils.data import sample_batch
from trojanzoo.utils.metric import mask_jaccard
from trojanzoo.utils.output import ansi, output_iter, prints

import torch
import numpy as np
import argparse
import os

import torch.utils.data


# use_mask: bool = True
# count_mask: bool = True
# ImageNet arguments from https://github.com/naiyeleo/ABS/blob/master/TrojAI_competition/round1/abs_pytorch_round1.py
# {
#     'samp_k': 2,
#     'defense_remask_lr': 0.4,
#     'defense_remask_epoch': 50,
#     'remask_weight': 50,
#     'max_troj_size': 400,
#     'top_n_neurons': 10,
# }


class ABS(ModelInspection):
    r"""Artificial Brain Stimulation proposed by Yingqi Liu
    from Purdue University in CCS 2019.

    It is a model inspection backdoor defense
    that inherits :class:`trojanvision.defenses.ModelInspection`.

    See Also:
        * paper: `ABS\: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation`_
        * code: https://github.com/naiyeleo/ABS

    .. _ABS\: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation:
        https://openreview.net/forum?id=YHWF1F1RBgF
    """
    name: str = 'abs'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--seed_data_num', type=int,
                           help='abs seed data number (default: -5)')
        group.add_argument('--mask_eps', type=float,
                           help='abs mask epsilon threshold (default: 0.01)')
        group.add_argument('--samp_k', type=int,
                           help='abs sample k multiplier in neuron sampling (default: 8)')
        group.add_argument('--same_range', action='store_true',
                           help='abs same range in neuron sampling (default: False)')
        group.add_argument('--n_samples', type=int,
                           help='abs n_samples in neuron sampling (default: 5)')
        group.add_argument('--top_n_neurons', type=int,
                           help='abs top-n neuron number in neuron sampling (default: 20)')
        group.add_argument('--max_troj_size', type=int,
                           help='abs max trojan trigger size in pixel number (default: 16)')
        group.add_argument('--remask_weight', type=float,
                           help='abs optimization norm loss weight (default: 500)')
        return group

    def __init__(self, seed_data_num: int = -5, mask_eps: float = 0.01,
                 samp_k: int = 8, same_range: bool = False,
                 n_samples: int = 5, top_n_neurons: int = 20,
                 max_troj_size: int = 16, remask_weight: float = 500.0,
                 defense_remask_lr: float = 0.1,
                 defense_remask_epoch: int = 1000,
                 **kwargs):
        super().__init__(defense_remask_epoch=defense_remask_epoch,
                         defense_remask_lr=defense_remask_lr,
                         cost=0.0, **kwargs)
        self.param_list['abs'] = ['seed_data_num', 'mask_eps',
                                  'samp_k', 'same_range',
                                  'n_samples', 'top_n_neurons',
                                  'max_troj_size', 'remask_weight']

        if seed_data_num < 0:
            seed_data_num = self.model.num_classes * abs(seed_data_num)
        self.seed_data_num = seed_data_num

        self.mask_eps = mask_eps
        # -----------Neural Sampling------------- #
        self.samp_k = samp_k
        self.same_range = same_range
        self.n_samples = n_samples
        self.top_n_neurons = top_n_neurons

        self.max_troj_size = max_troj_size
        self.remask_weight = remask_weight

        self.seed_data = self.get_seed_data()
        self.loader = [(self.seed_data['input'], self.seed_data['label'])]

    def get_mark_loss_list(self) -> tuple[torch.Tensor, list[float], list[float]]:
        print('sample neurons')
        all_ps = self.sample_neuron(self.seed_data['input'])
        print('find min max')
        self.neuron_dict = self.find_min_max(all_ps, self.seed_data['label'])

        format_str = self.serialize_format(layer='20s', neuron='5d', value='10.3f')
        # Output neuron dict information
        for label in range(self.model.num_classes):
            print('Class: ', output_iter(label, self.model.num_classes))
            for _dict in reversed(self.neuron_dict[label]):
                prints(format_str.format(**_dict), indent=4)
        print()
        print('optimize marks')
        return super().get_mark_loss_list(verbose=False)

    def optimize_mark(self, label: int, **kwargs) -> tuple[torch.Tensor, float]:
        format_dict = dict(layer='20s', neuron='5d', value='10.3f',
                           loss='10.3f', asr='8.3f', norm='8.3f')
        if not self.attack.mark.mark_random_pos:
            format_dict['jaccard'] = '5.3f'
            select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
        format_str = self.serialize_format(**format_dict)
        mark_best: torch.Tensor = torch.ones_like(self.attack.mark.mark)
        loss_best: float = 1e7
        asr_best: float = 0.0
        dict_best = {}
        for _dict in reversed(self.neuron_dict[label]):
            mark, loss = super().optimize_mark(label, loader=self.loader,
                                               verbose=False, **_dict)
            _dict['mark'] = mark.detach().cpu().clone().numpy()
            asr, _ = self.model._validate(get_data_fn=self.attack.get_data,
                                          keep_org=False, verbose=False)
            norm = float(mark[-1].flatten().norm(p=1))
            str_dict = dict(loss=loss, asr=asr, norm=norm, **_dict)
            if not self.attack.mark.mark_random_pos:
                overlap = mask_jaccard(self.attack.mark.get_mask(),
                                       self.real_mask,
                                       select_num=select_num)
                str_dict['jaccard'] = overlap
            prints(format_str.format(**str_dict), indent=4)
            if asr > asr_best:
                asr_best = asr
                mark_best = mark
                loss_best = loss
                dict_best = str_dict
        format_str = self.serialize_format(color='yellow', **format_dict)
        print()
        prints(format_str.format(**dict_best), indent=4)
        self.attack.mark.mark = mark_best
        return mark_best, loss_best

    def loss(self, _input: torch.Tensor, _label: torch.Tensor,
             layer: str, neuron: int, **kwargs) -> torch.Tensor:
        trigger_input = self.attack.add_mark(_input)
        feats = self.model.get_layer(trigger_input, layer_output=layer)
        feats = feats + (feats > 0) * feats    # sum up with relu
        vloss1 = feats[:, neuron].sum()
        vloss2 = feats.sum() - vloss1

        # if not use_mask:
        #     tvloss = total_variation(self.attack.mark.mark[:-1])
        #     ssim_loss = - self.ssim(X, _input)
        #     ssim_loss *= 10 if ssim_loss < -2 else 10000
        #     loss = -vloss1 + 1e-5 * vloss2 + 1e-3 * tvloss
        #     loss = 0.01 * loss + ssim_loss

        norm_loss = self.attack.mark.mark[-1].sum()
        mask_nz = float((self.attack.mark.mark[-1] > self.mask_eps).sum())
        if mask_nz > 1.2 * self.max_troj_size:
            norm_loss *= 2 * self.remask_weight
        elif mask_nz > self.max_troj_size:
            norm_loss *= self.remask_weight
        else:
            norm_loss *= 0.01
        return -vloss1 + 1e-4 * vloss2 + norm_loss

    # ---------------------------- Seed Data --------------------------- #
    def gen_seed_data(self) -> dict[str, np.ndarray]:
        r"""Generate seed data.

        Returns:
            dict[str, numpy.ndarray]:
                Seed data dict with keys ``'input'`` and ``'label'``.
        """
        torch.manual_seed(env['seed'])
        if self.seed_data_num % self.model.num_classes:
            raise ValueError(
                f'seed_data_num({self.seed_data_num:d}) % num_classes({self.model.num_classes:d}) should be 0.')
        seed_class_num: int = self.seed_data_num // self.model.num_classes
        x, y = [], []
        for _class in range(self.model.num_classes):
            class_set = self.dataset.get_dataset(mode='train', class_list=[_class])
            _input, _label = sample_batch(class_set, batch_size=seed_class_num)
            x.append(_input)
            y.append(_label)
        x = torch.cat(x).numpy()
        y = torch.cat(y).numpy()
        seed_data = {'input': x, 'label': y}
        seed_path = os.path.join(self.folder_path, f'seed_{self.seed_data_num}.npz')
        np.savez(seed_path, **seed_data)
        print('seed data saved at: ', seed_path)
        return seed_data

    def get_seed_data(self) -> dict[str, torch.Tensor]:
        r"""Get seed data. If npz file doesn't exist,
        call :meth:`gen_seed_data()` to generate.
        """
        seed_path = os.path.join(self.folder_path, f'seed_{self.seed_data_num}.npz')
        seed_data: dict[str, torch.Tensor] = {}
        seed_data_np = dict(np.load(seed_path)) if os.path.exists(seed_path) \
            else self.gen_seed_data()
        seed_data['input'] = torch.from_numpy(seed_data_np['input']).to(device=env['device'])
        seed_data['label'] = torch.from_numpy(seed_data_np['label']).to(device=env['device'], dtype=torch.long)
        return seed_data

    # -----------------------Neural Sample---------------------------- #

    def sample_neuron(self, _input: torch.Tensor) -> dict[str, torch.Tensor]:
        all_ps: dict[str, torch.Tensor] = {}
        layer_output = self.model.get_all_layer(_input, depth=2)
        for layer in layer_output.keys():
            if not layer.startswith('features.') and not layer.startswith('classifier.'):
                continue
            cur_layer_output: torch.Tensor = layer_output[layer].detach().cpu()  # (batch_size, C', H', W')
            channel_num: int = cur_layer_output.shape[1]  # channels

            h_t: torch.Tensor = cur_layer_output.expand([channel_num, self.n_samples] + [-1] * cur_layer_output.dim()
                                                        ).clone()
            # (C, n_samples, batch_size, C', H', W')

            vs = self.samp_k * torch.arange(self.n_samples, device=h_t.device, dtype=torch.float)
            if not self.same_range:
                maxes = cur_layer_output.max()
                vs *= float(maxes) / self.n_samples
            vs_shape = [1] * cur_layer_output.dim()
            vs_shape[0] = -1
            vs = vs.view(vs_shape)  # (n_samples, 1, 1, 1)
            # todo: might use parallel to avoid for loop (torch.Tensor.scatter?)
            for neuron in range(channel_num):
                h_t[neuron, :, :, neuron] = vs
            # todo: the shape is too large
            # result = self.model.get_layer(h_t.flatten(end_dim=2), layer_input=layer).detach().cpu()
            result = []
            for h in h_t:
                h: torch.Tensor
                h = h.to(device=env['device'])
                result.append(self.model.get_layer(h.flatten(end_dim=1), layer_input=layer
                                                   ).detach().cpu())
            result = torch.cat(result)

            result_shape = list(h_t.shape)[:3]
            result_shape.extend(list(result.shape)[1:])
            result = result.view(result_shape)
            all_ps[layer] = result  # (C, n_samples, batch_size, num_classes)
        return all_ps

    def find_min_max(self, all_ps: dict[str, torch.Tensor], _label: torch.Tensor
                     ) -> dict[int, list[dict[str, str | int | float]]]:
        neuron_dict: dict[int, list[dict[str, str | int | float]]]
        neuron_dict = {i: [] for i in range(self.model.num_classes)}
        _label = _label.cpu()
        for layer in all_ps.keys():
            ps = all_ps[layer]  # (C, n_samples, batch_size, num_classes)
            vs: torch.Tensor = ps[:, self.n_samples // 5:].amax(dim=1) \
                - ps[:, :self.n_samples // 5].amin(dim=1)  # (C, batch_size, num_classes)
            values, labels = vs.sort(dim=-1, descending=True)
            condition1 = labels[..., 0].eq(_label)  # exclude the ground-truth labels
            values = torch.where(condition1, values[..., 1] - values[..., 2],
                                 values[..., 0] - values[..., 1])  # (C, batch_size)
            labels = torch.where(condition1, labels[..., 1], labels[..., 0])  # (C, batch_size)

            mode_labels = labels.mode(keepdim=True)[0]  # (C, 1) The most frequent label
            mode_idx = labels.eq(mode_labels)  # (C, batch_size)
            mode_labels_counts = mode_idx.sum(dim=-1)  # (C)
            condition2 = mode_labels_counts.ge(self.seed_data_num * 0.75)
            idx_list = condition2.nonzero().flatten().tolist()
            idx_list = sorted(idx_list, key=lambda idx: float(values[idx][mode_idx[idx]].min()))[:self.top_n_neurons]
            for idx in idx_list:
                value = float(values[idx][mode_idx[idx]].min())
                neuron_dict[int(mode_labels[idx])].append(
                    {'layer': layer, 'neuron': int(idx), 'value': value})
        return neuron_dict

    @staticmethod
    def serialize_format(color: str = 'green', **kwargs: str) -> str:
        _str = ''
        for k, v in kwargs.items():
            _str += '{color}{k}{reset}: {{{k}:{v}}}    '.format(k=k, v=v, color=ansi[color], reset=ansi['reset'])
        return _str.removesuffix('    ')

    # ---------------------------------- Unused ------------------------------- #
    # def filter_img(self):
    #     h, w = self.dataset.data_shape[1:]
    #     mask = torch.zeros(h, w)
    #     mask[2:7, 2:7] = 1
    #     return mask.to(device=env['device'])

    # def nc_filter_img(self) -> torch.Tensor:
    #     h, w = self.dataset.data_shape[1:]
    #     mask = torch.ones(h, w)
    #     return mask.to(device=env['device'])
    #     # TODO: fix
    #     # mask = torch.zeros(h, w)
    #     # if self.use_mask:
    #     #     mask[math.ceil(0.25 * w): math.floor(0.75 * w), math.ceil(0.25 * h): math.floor(0.75 * h)] = 1
    #     # else:
    #     #     mask.add_(1)
