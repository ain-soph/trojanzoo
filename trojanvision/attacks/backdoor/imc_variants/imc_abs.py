#!/usr/bin/env python3

from trojanvision.attacks.backdoor.imc import IMC
from trojanvision.environ import env
from trojanzoo.utils import to_tensor

import torch
import numpy as np
import os


class IMC_ABS(IMC):
    name: str = 'imc_abs'

    def __init__(self, seed_num: int = -5, count_mask: bool = True,
                 samp_k: int = 1, same_range: bool = False, n_samples: int = 5,
                 **kwargs):
        super().__init__(**kwargs)

        self.seed_num: int = seed_num
        if self.seed_num < 0:
            self.seed_num = self.model.num_classes * abs(self.seed_num)
        self.count_mask: bool = count_mask

        # -----------Neural Sampling------------- #
        self.samp_k: int = samp_k
        self.same_range: bool = same_range
        self.n_samples: int = n_samples
        self.top_n_neurons: int = 20

        self.seed_data = self.load_seed_data()
        self.neuron_list = []

    def attack(self, epoch: int, **kwargs):
        super(IMC, self).attack(epoch, epoch_fn=self.epoch_fn, loss_fn=self.loss_fn, **kwargs)

    def epoch_fn(self, **kwargs):
        self.save()
        _input, _label = self.seed_data['input'], self.seed_data['label']
        all_ps = self.sample_neuron(_input)
        self.neuron_list: list[dict] = self.find_min_max(all_ps, _label)[0]
        self.optimize_mark()

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
        for layer in self.model.get_layer_name():
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
                h: torch.Tensor
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

    def abs_loss(self, layer_dict: dict[str, torch.Tensor], layer: str, neuron: int):
        feats = layer_dict[layer]
        vloss1 = feats[:, neuron].sum()
        vloss2 = feats.sum() - vloss1
        return -vloss1 + 1e-3 * vloss2

    def loss_fn(self, _input: torch.Tensor, _label: torch.Tensor, **kwargs) -> torch.Tensor:
        loss = self.model.loss(_input, _label)
        idx = 0
        for i in range(len(_label)):
            if _label[len(_label) - 1 - i] != self.target_class:
                break
            idx = len(_label) - 1 - i
        _input = _input[idx:]
        _label = _label[idx:]
        layer_dict = self.model.get_all_layer(_input)
        for sub_dict in self.neuron_list:
            layer = sub_dict['layer']
            neuron = sub_dict['neuron']
            loss -= 1e-5 * self.abs_loss(layer_dict, layer, neuron)
        return loss
